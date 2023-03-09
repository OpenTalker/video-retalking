#!/bin/bash          

# delete previous gh-pages
git branch -D gh-pages
git push origin :gh-pages

git checkout -b gh-pages
git rebase master
git reset HEAD

# make docs
./scripts/make_docs.sh

# Add docs
mv docs/_build/html/*.html .
git add *.html
mv docs/_build/html/*.js .
git add *.js
mv docs/_build/html/_static/ _static
git add _static

touch .nojekyll
git add .nojekyll

# Publish to gh-pages
git commit -m "docs"
git push origin gh-pages

git checkout master
