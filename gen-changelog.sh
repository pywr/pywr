#!/usr/bin/env bash
# Script to generate CHANGELOG.md from git commits
# Usage: ./gen-changelog.sh <release-tag>
#
# The script will generate a new CHANGELOG.md file with all the commits from v1.21.0 to the latest tag. These
# commits should follow the conventional commits format. The script will also append the older manually written
# changelog entries to the end of the new file.
git cliff -o CHANGELOG_TEMP.md --tag "$1" v1.20.1..
tail -n 557 CHANGELOG.md >> CHANGELOG_TEMP.md
mv CHANGELOG_TEMP.md CHANGELOG.md
