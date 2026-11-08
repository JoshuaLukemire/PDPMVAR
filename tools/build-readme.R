# Run from the package directory with Rscript tools/build-readme.R.
# Give pak an index of the installed packages for this build only.
repo <- tempfile("installed-packages-")
url  <- paste0("file://", repo)

installed <- installed.packages(fields = "NeedsCompilation")
fields    <- c("Package", "Version", "Depends", "Imports", "LinkingTo",
               "Suggests", "NeedsCompilation")
records   <- installed[!duplicated(installed[, "Package"]), fields, drop = FALSE]

for (type in unique(c("source", .Platform$pkgType))) {
  path <- sub("^file://", "", contrib.url(url, type = type))
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  write.dcf(records, file.path(path, "PACKAGES"))

  index <- gzfile(file.path(path, "PACKAGES.gz"), "wt")
  write.dcf(records, index)
  close(index)
}

options(repos = c(CRAN = url))
Sys.setenv(
  PKG_USE_BIOCONDUCTOR = "false",
  R_USER_CACHE_DIR = tempfile("readme-cache-")
)

devtools::build_readme(quiet = FALSE)
