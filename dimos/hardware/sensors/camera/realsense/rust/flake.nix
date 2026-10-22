{
  description = "RealSense camera native module for dimos";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dimos-realsense";

        # Keep this flake self-contained: shared SDK/configuration arrive through
        # one fixed-hash project source, rather than paths outside this directory.
        sharedArchive = pkgs.fetchurl {
          url = "https://codeload.github.com/dimensionalOS/dimos/tar.gz/8101c8db4a50043f1dbd0127fd00220ec9b5a37a";
          sha256 = "fcee2041d31c9368b77b803140f976b5a5d044f48f9423fea298afe372568361";
        };
        sharedSource = pkgs.runCommand "dimos-cdr-sdk-source" {
          nativeBuildInputs = [ pkgs.gnutar pkgs.gzip ];
        } ''
          mkdir -p $out
          tar -xzf ${sharedArchive} --strip-components=1 -C $out
        '';
        messages = import "${sharedSource}/native/message-source.nix" { inherit pkgs; };
        cameraSource = nix-filter.lib { root = ./.; exclude = [ "target" "build" "result" "__pycache__" ]; };
        src = pkgs.runCommand "dimos-realsense-cdr-source" { } ''
          mkdir -p $out/native/rust $out/build/message-source $out/dimos/hardware/sensors/camera/realsense/rust
          cp -r ${sharedSource}/native/rust/dimos-module $out/native/rust/
          cp -r ${sharedSource}/native/rust/dimos-module-macros $out/native/rust/
          cp -r ${sharedSource}/native/rust/dimos-tf-messages $out/native/rust/
          cp -r ${messages.source}/rust $out/build/message-source/rust
          cp -r ${cameraSource}/. $out/dimos/hardware/sensors/camera/realsense/rust/
        '';
        generated = crate2nix.tools.${system}.generatedCargoNix {
          inherit name src;
          cargoToml = "dimos/hardware/sensors/camera/realsense/rust/Cargo.toml";
        };

        needsLibrealsense = _: {
          buildInputs = [ pkgs.librealsense ];
          nativeBuildInputs = [ pkgs.pkg-config ];
        };
        sysOverrides = {
          realsense-sys = needsLibrealsense;
          dimos-realsense = needsLibrealsense;
        };

        ours = [ name "dimos-module" "dimos-module-macros" "dimos-tf-messages" ];
        callWith = mode: import generated {
          inherit pkgs;
          buildRustCrateForPkgs = cratePkgs:
            let build = cratePkgs.buildRustCrate.override {
                  # stabby-macros writes the builder's core count (NUM_JOBS) into its code; one value makes every machine's copy match.
                  defaultCrateOverrides = cratePkgs.defaultCrateOverrides // sysOverrides // { stabby-macros = _: { preConfigure = "export NIX_BUILD_CORES=1"; }; };
                };
            in crate: build (crate // pkgs.lib.optionalAttrs
              (mode != null && builtins.elem crate.crateName ours)
              ({
                release = false;
                extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-C" "debuginfo=0" ];
              } // pkgs.lib.optionalAttrs (mode == "lint") {
                useClippy = true;
                capLints = "forbid";
                extraRustcOpts =
                  (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" "-C" "debuginfo=0" ];
              }));
        };
        buildOf = called:
          if called ? rootCrate then called.rootCrate.build
          else called.workspaceMembers.${name}.build;
      in {
        packages.default = buildOf (callWith null);
        packages.${name} = self.packages.${system}.default;
        packages.lint = (buildOf (callWith "lint")).override {
          runTests = true;
          testCrateFlags = [ "--list" ];
        };
        checks.lint = self.packages.${system}.lint;

        packages.tests = (buildOf (callWith "test")).override { runTests = true; };
        checks.tests = self.packages.${system}.tests;

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt
                       pkgs.librealsense pkgs.pkg-config ];
        };
      });
}
