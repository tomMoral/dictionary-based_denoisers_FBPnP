import pandas as pd
import matplotlib.pyplot as plt


methods = {
    'DRUNet': 'DRUNet[pretrained=download]',
    'Synthesis': 'synthesis[accelerated=False,denoiser_accelerated=False,denoiser_n_layers=20,init_dual=True,n_epochs=50,n_layers=20,step_size_scaling=1.9]',
    'Analysis': 'Analysis[accelerated=False,denoiser_accelerated=False,denoiser_n_layers=20,init_dual=True,n_epochs=50,n_layers=20,step_size_scaling=1.9]',
    'Synthesis 1': 'synthesis[accelerated=False,denoiser_accelerated=False,denoiser_n_layers=20,init_dual=True,n_epochs=50,n_layers=1,step_size_scaling=1.9]',
    'Analysis 1': 'Analysis[accelerated=False,denoiser_accelerated=False,denoiser_n_layers=20,init_dual=True,n_epochs=50,n_layers=1,step_size_scaling=1.9]',
    'Synthesis (fast)': 'synthesis[accelerated=True,denoiser_accelerated=True,denoiser_n_layers=20,init_dual=True,n_epochs=50,n_layers=20,step_size_scaling=0.9]',
    'Analysis (fast)': 'Analysis[accelerated=True,denoiser_accelerated=True,denoiser_n_layers=20,init_dual=True,n_epochs=50,n_layers=20,step_size_scaling=0.9]',
}
reverse_methods = {v: k for k, v in methods.items()}

df = pd.read_parquet("benchopt_run_2025-02-25_00h01m51.parquet")
df.loc[:, 'solver_name'] = (
    df.apply(lambda g: g['solver_name'].replace(
        f",random_state={int(g['p_solver_random_state']) if not pd.isna(g['p_solver_random_state']) else ''}",
        ""), axis=1)
)

df_denoising = df.query('objective_task == "denoising"')
perf_denoising = df_denoising.query('objective_sigma == 0.1').groupby(['solver_name', 'objective_sigma'])[['objective_PSNR', 'objective_SSIM']].median()

perf_denoising['solver'] = perf_denoising['solver_name'].apply(lambda x: reverse_methods.get(x, x))
print(perf_denoising.set_index('solver').loc[list(reverse_methods.values())])

fig = plt.figure(figsize=(6.4, 3))
g = plt.GridSpec(nrows=2, ncols=2, height_ratios=[0.1, 0.9])
ax_psnr = fig.add_subplot(g[1, 0])
ax_res = fig.add_subplot(g[1, 1])
for i, (name, sn) in enumerate(methods.items()):
    print(name)
    df_solver = df.query(
        'objective_sigma == 0.001 and solver_name == @sn '
        'and objective_task == "blur"'
    )
    df_solver.groupby("objective_iter")['objective_psnr'].median().plot(ax=ax_psnr, color=f"C{i}")
    df_solver.groupby("objective_iter")['objective_residual'].median().plot(ax=ax_res, color=f"C{i}")
ax_res.set_yscale('log')
ax_legend = fig.add_subplot(g[0, :])
ax_legend.legend(
    [plt.Line2D([], [], c=f"C{i}") for i in range(len(methods))], methods,
    loc='center', ncols=4
)
ax_legend.set_axis_off()

fig.savefig("plot_cvg_pnp.pdf")
