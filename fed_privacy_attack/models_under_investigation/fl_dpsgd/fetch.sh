scp -pr ~/gcp_a100/FL5_dpsgd/checkpoints_3d_variation0/epoch114* .

mv epoch114* gcp_FL_noniid_dpsgd_noise0.5.h5

scp -pr edhlee@kristen0:~/Covid_March2023/FL5_dpsgd/checkpoints_3d_variation0/epoch136global.h5 .

mv epoch136* k0_FL_noniid_dpsgd_noise0.3.h5

