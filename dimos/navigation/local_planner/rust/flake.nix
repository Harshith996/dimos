{
  description = "SE(2) local planner native module for dimos";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dimos-local-planner";

        moduleSource = nix-filter.lib { root = ./.; exclude = [ "target" "build" "result" "__pycache__" ]; };
        sdkArchive = pkgs.fetchurl {
          url = "https://codeload.github.com/dimensionalOS/dimos/tar.gz/937e53e824b7a26bd52421ba46687d211f9e1e7b";
          sha256 = "e7db5c2d0cc13a9e6dd76d90e4ee444fa99840f9ab0bf4c16139ba12673b1109";
        };
        sdkSource = pkgs.runCommand "dimos-local-planner-sdk-source" {
          nativeBuildInputs = [ pkgs.gnutar pkgs.gzip ];
        } ''
          mkdir -p $out
          tar -xzf ${sdkArchive} --strip-components=1 -C $out
        '';
        messageSource = (import "${sdkSource}/native/message-source.nix" { inherit pkgs; }).source;
        # Standalone flake builds resolve explicit source artifacts, not checkout parents.
        src = pkgs.runCommand "local-planner-cdr-source" { } ''
          cp -r ${moduleSource} $out
          chmod -R u+w $out
          mkdir -p $out/sdk
          cp -r ${sdkSource}/native/rust/dimos-module ${sdkSource}/native/rust/dimos-module-macros ${sdkSource}/native/rust/dimos-message-adapters $out/sdk/
          cp -r ${messageSource}/rust $out/messages
          substituteInPlace $out/sdk/dimos-message-adapters/Cargo.toml \
            --replace-fail '../../../build/message-source/rust' '../../messages'
          substituteInPlace $out/Cargo.toml \
            --replace-fail '../../../../native/rust/dimos-module' 'sdk/dimos-module' \
            --replace-fail '../../../../native/rust/dimos-message-adapters' 'sdk/dimos-message-adapters' \
            --replace-fail '../../../../build/message-source/rust' 'messages'
        '';

        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

        ours = [ name "dimos-module" "dimos-module-macros" "dimos-message-adapters" ];
        callWith = mode: import generated {
          inherit pkgs;
          rootFeatures = if mode == null then [ "module" ] else [ "module" "test-bodies" ];
          buildRustCrateForPkgs = cratePkgs: crate:
            # stabby-macros writes the builder's core count (NUM_JOBS) into its code; one value makes every machine's copy match.
            (cratePkgs.buildRustCrate.override {
              defaultCrateOverrides = cratePkgs.defaultCrateOverrides // { stabby-macros = _: { preConfigure = "export NIX_BUILD_CORES=1"; }; };
            }) (crate // pkgs.lib.optionalAttrs
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

        rustTools = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
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

        devShells.default = pkgs.mkShell { packages = rustTools; };
      });
}
