import sys
import torch
import faiss

print(f"--- Environment Check ---")
print(f"Python Version: {sys.version.split()[0]}")

# 1. Check PyTorch and GPU
try:
    print(f"\n[1/3] Checking PyTorch...")
    gpu_available = torch.cuda.is_available()
    print(f"      Torch Version: {torch.__version__}")
    print(f"      GPU Available: {gpu_available}")
    if gpu_available:
        print(f"      GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("      WARNING: Torch cannot see your GPU. 'bioel' might run very slowly.")
except ImportError:
    print("      ERROR: PyTorch is not installed correctly.")

# 2. Check Faiss (The tricky dependency)
try:
    print(f"\n[2/3] Checking Faiss (The problematic package)...")
    # Create a simple index to ensure the binary is compatible
    d = 64                           # dimension
    nb = 1000                        # database size
    xb = faiss.randn((nb, d), 1234)
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    print(f"      Success: Faiss created an index with {index.ntotal} vectors.")
    
    # Optional: Check if Faiss sees GPU (if you installed faiss-gpu)
    try:
        res = faiss.StandardGpuResources()
        print("      Success: Faiss can access GPU resources.")
    except AttributeError:
        print("      Note: StandardGpuResources not found. You might be running CPU-only Faiss.")
    except Exception as e:
        print(f"      Warning: Faiss GPU check failed ({e})")

except ImportError:
    print("      ERROR: Faiss is not installed or the version is incompatible.")
except Exception as e:
    print(f"      ERROR: Faiss crashed during initialization: {e}")

# 3. Check BioEL Imports
try:
    print(f"\n[3/3] Checking BioEL...")
    import bioel
    from bioel.model import BioEL_Model
    from bioel.ontology import BiomedicalOntology
    print(f"      Success: BioEL imported correctly.")
    print(f"      Location: {bioel.__file__}")
except ImportError as e:
    print(f"      ERROR: Could not import BioEL modules. {e}")
except Exception as e:
    print(f"      ERROR: BioEL crashed on import: {e}")

print("\n--- Test Finished ---")