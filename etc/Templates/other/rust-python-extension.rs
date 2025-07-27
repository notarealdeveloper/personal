/* 
    #!/bin/bash

    project="pyrs"
    mkdir -pv "$project"
    pushd "$project"
    python -m venv .env
    source .env/bin/activate
    pip install maturin

    # get a minimal working project
    maturin init

    # hack hack hack
    # add whatever functions you want

    # this builds and installs into the virtual env
    maturin develop

    # optional, to get ipython
    pip install --upgrade pip
    pip install ipython
    hash -r

    # now you can do this:
    # $ python
    # >>> import pyrs
    # >>> pyrs.sum_as_string(5, 20)
*/

