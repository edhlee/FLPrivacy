import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras import mixed_precision
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized, dp_optimizer_keras_vectorized
from sklearn.utils.class_weight import compute_sample_weight
import aux_models as aux_models
import tensorflow_addons as tfa
import pickle
import augmentation_helpers as augmentation_helpers

print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

tf.keras.backend.clear_session()
tf.compat.v1.enable_eager_execution()

# Mixed precision policy
use_fp16 = sys.argv[3].lower() == 'true'
policy_name = 'mixed_float16' if use_fp16 else 'float32'
policy = mixed_precision.Policy(policy_name)
tf.keras.mixed_precision.set_global_policy(policy)

# Constants
initial_learning_rate = 0.001
GLOBAL_OFFSET = 50
THRESHOLD_GOLD = 0.8
RESIZE_DIM_256 = (256, 256)
RESIZE_DIM_224 = (224, 224)
NUM_SITES = 21 

# Script arguments
global_test_index = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
global_offset_gold = int(sys.argv[4])
threshold_gold = float(sys.argv[5])
use_all_data = sys.argv[6].lower() == 'true'
global_starting_epoch = int(sys.argv[7])
global_client_idx = int(sys.argv[8])
global_num_clients = int(sys.argv[9])
global_num_local_epochs = int(sys.argv[10])
use_iid_data = sys.argv[11].lower() == 'true'
global_model_id = int(sys.argv[12])
model_subdir = str(sys.argv[13])
fraction_of_train_used = float(sys.argv[14])

np.random.seed(42)

# Data loading and processing
def load_data(site_range, file_prefix, data_type):
    data_x_all, data_y_all = [], []
    for site_idx in range(site_range):
        data = np.load(f"{file_prefix}/CovidData2022_cohort{site_idx}{data_type}.npy", allow_pickle=True)
        data_x = data[f"c{site_idx}_x"].copy()
        data_y = data[f"c{site_idx}_y"].copy()
        if site_idx == 0:
            data_x_all, data_y_all = data_x.copy(), np.zeros((len(data_y)), dtype=np.uint32)
        else:
            data_x_all = np.concatenate((data_x_all, data_x), axis=0)
            data_y_all = np.concatenate((data_y_all, np.ones((len(data_y)), dtype=np.uint32) * site_idx), axis=0)
    return data_x_all, data_y_all

train_x_all, train_y_all = load_data(NUM_SITES, "data", "train")
val_x_all, val_y_all = load_data(NUM_SITES, "data", "test")

# Randomize and select a fraction of training data
idx = np.random.permutation(len(train_x_all))
train_x = train_x_all[idx][:int(fraction_of_train_used * len(train_x_all))]
train_y = train_y_all[idx][:int(fraction_of_train_used * len(train_y_all))]



# Print data lengths for verification
print(f"Train data length: {len(train_x)}, Validation data length: {len(val_y_all)}")

weights = compute_sample_weight(class_weight='balanced', y=train_y)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, weights))
test_dataset = tf.data.Dataset.from_tensor_slices((val_x_all, val_y_all))


# Example usage
TRAIN_LENGTH = len(train_x)
TEST_LENGTH = len(val_x_all)



BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // (BATCH_SIZE)
global_offset = global_offset_gold

train_dataset_ = train_dataset.shuffle(BUFFER_SIZE, seed=42)
train_dataset_ = train_dataset_.map(augmentation_helpers.process_tf, 
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(BATCH_SIZE)
test_dataset_ = test_dataset.map(augmentation_helpers.test_map).batch(BATCH_SIZE)




train_loss_results = []
train_accuracy_results = []
num_epochs = global_num_local_epochs
steps_per_epoch = STEPS_PER_EPOCH

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=STEPS_PER_EPOCH // 4,
    decay_rate=0.1,
    staircase=True
)

# Optimizer setup
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
if use_fp16:
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

# Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Model setup
model = aux_models.i3d_v2()
model.compile(optimizer, loss)
model.load_weights("models_under_investigation/" + str(model_subdir), skip_mismatch=True, by_name=True)

# Metrics
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
f1_metric = tfa.metrics.F1Score(num_classes=NUM_SITES)
f1_metric_weighted = tfa.metrics.F1Score(num_classes=NUM_SITES, average="weighted")

# Evaluation function
def evaluate_step(model, val_dataset, test_accuracy, f1_metric, f1_metric_weighted, num_steps):
    num_eval_steps = num_steps // BATCH_SIZE
    for (batch, (x, z)) in enumerate(val_dataset.take(num_eval_steps)):
        logits = model(x, training=False)
        test_accuracy.update_state(z, logits)
        z2 = tf.one_hot(z.numpy(), depth=21)
        f1_metric.update_state(z2, logits.numpy())
        f1_metric_weighted.update_state(z2, logits.numpy())
    return f1_metric.result().numpy(), f1_metric_weighted.result().numpy()

# Training and evaluation loop
epochs, f1_scores, f1_scores_weighted = [], [], []
for i in range(20):
    f1_score, f1_score_weighted = evaluate_step(model, test_dataset_, test_accuracy, f1_metric, f1_metric_weighted, TEST_LENGTH)
    with open("evals_sites/configID_" + str(global_model_id) + "_sites.csv", 'a') as fd:
        fd.write(f"epoch: {i+1} model config: {global_test_index} f1_weighted: {f1_score_weighted}\n")
    f1_metric.reset_state()
    f1_metric_weighted.reset_state()
    epochs.append(i)
    f1_scores.append(f1_score)
    f1_scores_weighted.append(f1_score_weighted)
    model.fit(train_dataset_, epochs=1, steps_per_epoch=STEPS_PER_EPOCH)

# Save model and results
model.save("checkpoints_attackers/attacker_id" + str(global_model_id) + ".h5", include_optimizer=False)
all_data = {
    'epochs': np.array(epochs),
    'f1_weighted': np.array(f1_scores_weighted),
    'f1': np.array(f1_scores),
    'fraction': np.array(fraction_of_train_used)
}
with open("evals_sites_npy/configID_" + str(global_model_id) + "_sites.npy", 'wb') as filehandle:
    pickle.dump(all_data, filehandle, protocol=4)










