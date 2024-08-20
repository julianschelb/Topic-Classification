# ===========================================================================
#                            Helpers for working with GPUs
# ===========================================================================

import torch


def listAvailableGPUs():
    num_gpus = torch.cuda.device_count()
    gpu_info_list = []
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        gpu_info = {
            'Index': i,
            'Name': gpu.name,
            'Memory (MiB)': gpu.total_memory / (1024 ** 2),
            'Compute Capability': f'{gpu.major}.{gpu.minor}'
        }
        gpu_info_list.append(gpu_info)

        print(f'GPU {i}:')
        print(f'  Name: {gpu.name}')
        print(f'  Memory: {gpu.total_memory / (1024 ** 2):.2f} MiB')
        print(f'  Compute Capability: {gpu.major}.{gpu.minor}')
        print()

    return gpu_info_list



if __name__ == "__main__":
    # Call the function to list the available GPUs and their details:
    gpu_info_list = listAvailableGPUs()
