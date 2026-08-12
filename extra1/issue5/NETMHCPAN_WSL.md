# NetMHCpan 4.1b through WSL

This project can call the NetMHCpan installation in the `Ubuntu` WSL distro
from the Windows `my_pytorch` Python environment. The runner translates
Windows drive paths to `/mnt/<drive>/...` and converts project HLA notation
such as `HLA-A*02:01` to NetMHCpan notation `HLA-A02:01`.

Human:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_predictors.py netmhcpan `
  --wsl-distro Ubuntu `
  --executable /home/hitomi/netMHCpan-4.1/netMHCpan `
  --manifest results\issue5_general_pmhc\queries\human_netmhcpan_manifest.csv `
  --metadata results\issue5_general_pmhc\raw_outputs\human_netmhcpan.metadata.json
```

Mouse:

```powershell
E:\ancd\envs\my_pytorch\python.exe extra\issue5\run_predictors.py netmhcpan `
  --wsl-distro Ubuntu `
  --executable /home/hitomi/netMHCpan-4.1/netMHCpan `
  --manifest results\issue5_general_pmhc\queries\mouse_netmhcpan_manifest.csv `
  --metadata results\issue5_general_pmhc\raw_outputs\mouse_netmhcpan.metadata.json
```

Existing output files are reused. Add `--force` only when every output in the
selected manifest must be recomputed.
