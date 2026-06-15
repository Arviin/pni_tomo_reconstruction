import astra

print("ASTRA imported successfully.")
print("ASTRA version:", getattr(astra, "__version__", "unknown"))

try:
    print("CUDA available:", astra.astra.use_cuda())
except Exception as exc:
    print("Could not check CUDA availability.")
    print(exc)