import pathlib

SRC_DIR = pathlib.Path(__file__).absolute().parent
REPO_DIR = SRC_DIR.parent
DATA_DIR = REPO_DIR / "data"
OUTPUT_DIR = REPO_DIR / "output"
FIG_DIR = REPO_DIR / "figs"
FIGURES_PATH = FIG_DIR

LEGACY_PATH = REPO_DIR / "legacy"
LEGACY_DATA_PATH = LEGACY_PATH / "data"

LABELLED_DATA_DIR = DATA_DIR / "labelled_data" / "strict_labels"

LIBRARIES_IO_DIR = pathlib.Path("~/datasets/libraries-1.6.0-2020-01-12")
PROJ_REPOS_PATH = LIBRARIES_IO_DIR / "fixed_projects_with_repository_fields-1.6.0-2020-01-12.csv"
# The chunks are extracted from project_repos
PROJ_REPOS_CHUNKS_QA_DIR = DATA_DIR / "project_repo_chunks_for_qa"

LIBRARIES_IO_REPOSITORIES_PATH = LIBRARIES_IO_DIR / "repositories-1.6.0-2020-01-12.csv"
LIBRARIES_IO_REPOSITORIES_FIXED_PATH = LIBRARIES_IO_DIR / "repositories-1.6.0-2020-01-12_fixed.csv"
LIBRARIES_IO_PROJECT_WITH_REPO_PATH = LIBRARIES_IO_DIR / "projects_with_repository_fields-1.6.0-2020-01-12.csv"
LIBRARIES_IO_PROJECT_WITH_REPO_FIXED_PATH = LIBRARIES_IO_DIR / "projects_with_repository_fields-1.6.0-2020-01-12_fixed.csv"
LIBRARIES_IO_PROJECT_WITH_REPO_Ecos13_PATH = LIBRARIES_IO_DIR / "projects_with_repository_fields-1.6.0-2020-01-12_Ecos13.csv"
LIBRARIES_IO_PROJECT_PATH = LIBRARIES_IO_DIR / "projects-1.6.0-2020-01-12.csv"
LIBRARIES_IO_VERSIONS_PATH = LIBRARIES_IO_DIR / "versions-1.6.0-2020-01-12.csv"
LIBRARIES_IO_TAGS_PATH = LIBRARIES_IO_DIR / "tags-1.6.0-2020-01-12.csv"
