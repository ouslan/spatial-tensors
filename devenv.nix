{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
{
  # --- 1. Global Base Configuration ---
  env = {
    PROJECT_NAME = "spatial-tensors";
  };

  dotenv.enable = true;

  # We remove the imports = [ ./devenv ] because we are not creating a subfolder structure for now,
  # unless it's strictly required by the pattern. The user said "derived from the template".
  # Let's just keep it simple and put everything in devenv.nix first.

  scripts.hello.exec = ''
    echo "Welcome to spatial-tensors development environment!"
  '';

  # --- 2. Profile Definitions ---
  profiles = {
    dev.module = { config, ... }: {
      env = {
        UV_SYSTEM_PYTHON = "0";
      };

      packages = with pkgs; [
        git
        git-cliff
        opencommit
        jupyter
        nixpkgs-fmt
        # Adding spatial-tensors specific system deps if any were evident. 
        # pyproject.toml has geopandas (requires gdal, proj, geos) and pysal.
        # Normally uv handles these via wheels, but for Nix we might need them.
        gdal
        proj
        geos
      ];

      enterShell = ''
        echo "Spatial Tensors Dev Env Loaded"
        uv sync
      '';

      enterTest = ''
        echo "Running tests..."
        # Add actual test command here if available (e.g., pytest)
        uv run pytest
      '';
    };

    container-build.module = {
      env = {
        UV_SYSTEM_PYTHON = "1";
      };

      packages = with pkgs; [
        docker
        docker-buildx
        podman
        skopeo
      ];

      languages.python.uv.sync.enable = false;
      languages.python.lsp.enable = false;
    };
  };
}
