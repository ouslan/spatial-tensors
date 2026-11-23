{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{

  # https://devenv.sh/packages/
  packages = [
    pkgs.python312
    pkgs.stdenv.cc.cc.lib
  ];

}
