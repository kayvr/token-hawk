# TokenHawk CLI

CLI interface to TokenHawk. Requires.

# Build Instructions

Ensure you're recursively cloned the repository and all it's submodules.

## Preparing Dawn

```
$ cd <token-hawk-directory>
$ git submodule update --init --recursive
```

In order to build Dawn, you'll need Google's `depot_tools`. Here's some instructions to get you started:

```
$ cd /tmp
$ git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
$ export PATH=/tmp/depot_tools:$PATH
$ cd <token-hawk-directory>
$ cd cli/dawn
$ cp scripts/standalone.gclient .gclient
$ gclient sync
$ gn gen out/Release
$ autoninja -C out/Release
```

## Building TokenHawk

Compared to dawn, TokenHawk is relatively straight forward:
