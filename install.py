import subprocess

libs_to_check = [
    'torch',
    'matplotlib',
    'seaborn',
    'tensorflow',
    'torch.cuda.amp',
    'Plotter',
    'EfficientNetB0',
    'torch.optim',
    'sklearn',
    'EfficientNetModel',
    'DataLoaderClass',
    'timm',
    'keras',
    'numpy',
    'torchvision',
    'PIL',
]

def check_installation(lib):
    try:
        __import__(lib)
        print(f"{lib} já está instalado.")
        return True
    except ImportError:
        print(f"{lib} não está instalado.")
        return False

def install_library(lib):
    subprocess.call(['pip', 'install', lib])

def main():
    for lib in libs_to_check:
        if not check_installation(lib):
            install_library(lib)

if __name__ == "__main__":
    main()
