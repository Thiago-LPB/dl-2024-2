{ pkgs ? import <nixpkgs> {config = {allowUnfree = true; cudaSupport = true;};} }:
let
envWithScript = script:
(pkgs.buildFHSEnv {
  name = "python-env";
  targetPkgs = pkgs: (with pkgs; [
    (python312.withPackages(ps: with ps; [pip virtualenv jupyter-core pyqt6]))
    cudatoolkit
    zlib
    gdb
    valgrind
    graphviz
  ]);
  multiPkgs = pkgs: (with pkgs; [
    udev
    alsa-lib
  ]);
  runScript = "${pkgs.writeShellScriptBin "runScript" (''
              set -e
              test -d .nix-venv || ${pkgs.python312.interpreter} -m venv .nix-venv
              export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudatoolkit}
              export LD_LIBRARY_PATH=${pkgs.zlib}/lib/:${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.python312Packages.tkinter}/lib/:$LD_LIBRARY_PATH
              export CUDA_PATH=${pkgs.cudatoolkit}:${pkgs.cudaPackages.cudnn}
              export EXTRA_CCFLAGS="-I/usr/include"
              export PYTHONPATH=./.nix-venv/bin/
              echo "Develop system"
              source .nix-venv/bin/activate
              set +e
            ''
            + script)}/bin/runScript";
}).env;
in {
  devShell = envWithScript "zsh";
}
