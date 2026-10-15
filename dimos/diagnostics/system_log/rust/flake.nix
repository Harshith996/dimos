{
  description = "system_log native module for DimOS: machine health, a ring dumped on trouble, and dimos' logs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # The root workspace (this crate is a member), so crate2nix reads the shared Cargo.lock.
        src = pkgs.lib.fileset.toSource {
          root = ../../../..;
          fileset = pkgs.lib.fileset.unions [
            ../../../../Cargo.lock
            ../../../../Cargo.toml
            ../../../../dimos/experimental/memory/rust
            ../../../../native/rust/dimos-module
            ../../../../native/rust/dimos-module-macros
            ../../../../dimos/mapping/ray_tracing/rust
            ../../../../dimos/mapping/ray_tracing/rust/py
            ../../../../dimos/navigation/global_planner/mls_planner/rust
            ../../../../dimos/navigation/global_planner/mls_planner/rust/py
            ../../../../dimos/hardware/sensors/lidar/livox/rust
            ../../../../dimos/hardware/sensors/lidar/pointlio/rust
            ../../../../dimos/hardware/sensors/lidar/virtual_mid360
            ../../../../dimos/hardware/sensors/camera/v4l2/rust
            ../../../../dimos/diagnostics/system_log/rust
            ../../../../examples/native-modules/rust
          ];
        };

        generatedCargoNix = crate2nix.tools.${system}.generatedCargoNix {
          name = "system-log";
          inherit src;
        };
      in {
        packages.default = (import generatedCargoNix { inherit pkgs; }).workspaceMembers."dimos-system-log".build;
      });
}
