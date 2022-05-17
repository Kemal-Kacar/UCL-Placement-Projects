import nilearn
from nilearn.plotting import plot_anat
import matplotlib_inline
from nipype.interfaces.fsl import BET

plot_anat("mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz", title='original',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)

skullstrip = BET(in_file="/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
                 out_file="/output/T1w_nipype_bet.nii.gz",
                 mask=True)
res = skullstrip.run()
