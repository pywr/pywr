# Deploy to pypi
twine upload dist/*

# Deploy to anaconda
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  anaconda -t ${CONDA_UPLOAD_TOKEN} upload -u pywr ${HOME}/miniconda3/conda-bld/osx-64/*.tar.bz2 --force
else
  anaconda -t ${CONDA_UPLOAD_TOKEN} upload -u pywr ${HOME}/miniconda3/conda-bld/linux-64/*.tar.bz2 --force
fi

