# Dataset backends: benchmark_operator, vlasov_maxwell_gkeyll, fusion_real_lhd, d3d_real, fusion_real_tcv.

from prism_ad.datasets.benchmark_operator import generate_benchmark_operator
from prism_ad.datasets.vlasov_maxwell_gkeyll import generate_vm_gkeyll
from prism_ad.datasets.fusion_real_lhd import generate_lhd
from prism_ad.datasets.d3d_real import generate_d3d
from prism_ad.datasets.fusion_real_tcv import generate_tcv

DATASET_REGISTRY = {
    "benchmark_operator": generate_benchmark_operator,
    "vlasov_maxwell_gkeyll": generate_vm_gkeyll,
    "fusion_real_lhd": generate_lhd,
    "d3d_real": generate_d3d,
    "fusion_real_tcv": generate_tcv,
}
