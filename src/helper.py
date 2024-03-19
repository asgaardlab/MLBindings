import base64
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pickle
import yaml
from IPython.core.display import display, HTML

try:
    from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
except Exception:
    print("import facets_overview error!")
from scipy import stats

import cliffdelta
import commonpath

REPO_STARS_THRESHOLD = 1000
EXCLUDE_ECOSYSTEMS = [
    "Homebrew",
    "Conda",
    "Bower",
    "Wordpress",
]

FIG_SIZE = (10, 4)
FIG_SIZE_WIDE = (12, 6)

PICKUP_PLATFORMS = sorted([
    "NPM",  # js
    "Packagist",
    "Pypi",
    "NuGet",
    "Maven",
    "Rubygems",
    "CocoaPods",
    "CPAN",
    "Cargo",
    "Clojars",
    "CRAN",
    "Hackage",
    "Pub",
], key=lambda x: x.lower())

PLATFORM_NAME_FOR_SHOWN = {
    "NPM": "npm",
    "Pypi": "PyPI",
}

PICKUP_PLATFORMS_FOR_SHOWN = [PLATFORM_NAME_FOR_SHOWN[n] if n in PLATFORM_NAME_FOR_SHOWN else n for n in
                              PICKUP_PLATFORMS]

DTYPES_PROJECTS_WITH_REPO = {
    "ID": int,
    "Platform": str,
    "Name": str,
    "Created Timestamp": str,
    "Updated Timestamp": str,
    "Description": str,
    "Keywords": str,
    "Homepage URL": str,
    "Versions Count": int,
    "SourceRank": int,
    "Latest Release Publish Timestamp": str,
    "Latest Release Number": str,
    # "Package Manager ID": int,
    "Dependent Projects Count": float,
    "Language": str,
    "Status": str,
    "Last synced Timestamp": str,
    "Dependent Repositories Count": int,
    # "Repository ID": int,
    "Repository Host Type": str,
    "Repository Name with Owner": str,
    "Repository Description": str,
    "Repository Fork": bool,
    "Repository Created Timestamp": str,
    "Repository Updated Timestamp": str,
    "Repository Last pushed Timestamp": str,
    "Repository Homepage URL": str,
    # "Repository Size": int,
    # "Repository Stars Count": int,
    "Repository Language": str,
    "Repository Issues enabled": bool,
    "Repository Wiki enabled": bool,
    "Repository Pages enabled": bool,
    # "Repository Forks Count": int,
    "Repository Mirror URL": str,
    # "Repository Open Issues Count": int,
    "Repository Default branch": str,
    # "Repository Watchers Count": int,
    "Repository UUID": str,
    "Repository Fork Source Name with Owner": str,
    "Repository License": str,
    # "Repository Contributors Count": int,
    "Repository Readme filename": str,
    "Repository Changelog filename": str,
    "Repository Contributing guidelines filename": str,
    "Repository License filename": str,
    "Repository Code of Conduct filename": str,
    "Repository Security Threat Model filename": str,
    "Repository Security Audit filename": str,
    "Repository Status": str,
    "Repository Last Synced Timestamp": str,
    # "Repository SourceRank": int,
    "Repository Display Name": str,
    "Repository SCM type": str,
    "Repository Pull requests enabled": bool,
    "Repository Logo URL": str,
    "Repository Keywords": str,
}

DTYPES_VERSIONS = {
    "ID": int,
    "Platform": str,
    "Project Name": str,
    "Project ID": int,
    "Number": str,
    "Published Timestamp": str,
    "Created Timestamp": str,
    "Updated Timestamp": str,
}

DTYPES_REPO = {
    "ID": int,
    "Host Type": str,
    "Name with Owner": str,
    "Description": str,
    "Fork": bool,
    "Created Timestamp": str,
    "Updated Timestamp": str,
    "Last pushed Timestamp": str,
    "Homepage URL": str,
    # "Size": int,
    # "Stars Count": int,
    "Language": str,
    "Issues enabled": bool,
    "Wiki enabled": bool,
    # "Pages enabled": bool,
    # "Forks Count": int,
    "Mirror URL": str,
    # "Open Issues Count": int,
    "Default branch": str,
    # "Watchers Count": int,
    "UUID": str,
    "Fork Source Name with Owner": str,
    "License": str,
    # "Contributors Count": int,
    "Readme filename": str,
    "Changelog filename": str,
    "Contributing guidelines filename": str,
    "License filename": str,
    "Code of Conduct filename": str,
    "Security Threat Model filename": str,
    "Security Audit filename": str,
    "Status": str,
    "Last Synced Timestamp": str,
    "SourceRank": int,
    "Display Name": str,
    "SCM type": str,
    # "Pull requests enabled": bool,
    "Logo URL": str,
    "Keywords": str,
}

PATH_WITH_TYPES = {
    # commonpath.LIBRARIES_IO_PROJECT_WITH_REPO_FIXED_PATH: DTYPES_PROJECTS_WITH_REPO,
    # commonpath.COLLECTED_LIBS_PATH: DTYPES_PROJECTS_WITH_REPO,
    commonpath.PROJ_REPOS_PATH: DTYPES_REPO,
    # commonpath.COLLECTED_REPOS_PATH: DTYPES_REPO,
    # commonpath.LIBRARIES_IO_VERSIONS_PATH: DTYPES_VERSIONS,
    # commonpath.COLLECTED_VERSIONS_PATH: DTYPES_VERSIONS,
}

DEFAULT_USED_COLS = {
    commonpath.PROJ_REPOS_PATH: [
        'ID',
        'Platform',
        'Name',
        'Description',
        'Keywords',
        'Licenses',
        "Homepage URL",
        'Repository URL',
        "Repository Homepage URL",
        'Versions Count',
        'Dependent Projects Count',
        'Dependent Repositories Count',
        'Status',
        'Repository ID',
        'Repository License',
        'Repository Host Type',
        'Repository Name with Owner',
        'Repository Description',
        'Repository Keywords',
        'Repository Fork?',
        'Repository Stars Count',
        'Repository Status',
        # 'Repository Homepage URL',
    ],
}

class FileAlreadyExist(Exception):
    pass


def formatIDToStr(id_instance):
    if np.isnan(id_instance) or isinstance(id_instance, str):
        return id_instance
    return str(int(id_instance))


def formatIDColumnsToStr(df):
    cols = [c for c in df.columns if ("ID" in c and "IDs" not in c)]
    for c in cols:
        df[c] = df[c].apply(formatIDToStr)
    return df


def toDatetime(series):
    if series.values[0].endswith(" UTC"):
        return pandas.to_datetime(series.apply(lambda x: x[:-4]), format='%Y-%m-%d %H:%M:%S')
    else:
        return pandas.to_datetime(series, format='%Y-%m-%d %H:%M:%S')


def valuesCount(df):
    c = df.value_counts(dropna=False)
    p = df.value_counts(dropna=False, normalize=True) * 100
    return pandas.concat([c, p], axis=1, keys=['counts', '%'])


def facets_overview(df, output_html_path):
    # Add the path to the feature stats generation code.
    # import sys
    # sys.path.insert(0, '/Users/poudel/Softwares/facets/facets_overview/python/')

    # Create the feature stats for the datasets and stringify it.
    import base64

    gfsg = GenericFeatureStatisticsGenerator()
    proto = gfsg.ProtoFromDataFrames(
        [{'name': 'data', 'table': df}]
    )
    # proto = gfsg.ProtoFromDataFrames(
    #     [{"name": name, 'table': g} for name, g in df.groupby("Library Name")]
    # )

    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")

    # Display the facets overview visualization for this data
    from IPython.core.display import display, HTML

    HTML_TEMPLATE = """
          <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
          <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html" >
          <facets-overview id="elem"></facets-overview>
          <script>
            document.querySelector("#elem").protoInput = "{protostr}";
          </script>"""
    html_str = HTML_TEMPLATE.format(protostr=protostr)
    html = HTML(html_str)

    display(html)  # this works in google colab but not in my computer.

    # the saved output.html works fine in my local computer.
    with open(output_html_path, 'w') as fo:
        fo.write(html_str)


def facets_dive(df, output_html_path):
    # Create the feature stats for the datasets and stringify it.
    jsonstr = df.to_json(orient='records')
    HTML_TEMPLATE = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
            <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
            <facets-dive id="elem"></facets-dive>
            <script>
              var data = {jsonstr};
              document.querySelector("#elem").data = data;
            </script>"""
    html_str = HTML_TEMPLATE.format(jsonstr=jsonstr)
    html = HTML(html_str)
    display(html)  # this works in google colab but not in my computer.

    # the saved output.html works fine in my local computer.
    with open(output_html_path, 'w') as fo:
        fo.write(html_str)



def dumpPickle(data, filepath):
    with open(filepath, 'wb') as file:
        return pickle.dump(data, file)


def loadPickle(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def read_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def savefig(fig, name, strip_title=True):
    if strip_title:
        p_title = fig.suptitle('').get_text()
        if len(fig.axes) == 1:
            a_title = fig.axes[0].get_title()
            fig.axes[0].set_title('')

    fig.savefig(
        (commonpath.FIGURES_PATH / '{}.pdf'.format(name)).as_posix(),
        bbox_inches='tight'
    )
    fig.savefig(
        (commonpath.FIGURES_PATH / '{}.png'.format(name)).as_posix(),
        bbox_inches='tight'
    )

    if strip_title:
        fig.suptitle(p_title)
        if len(fig.axes) == 1:
            fig.axes[0].set_title(a_title)


def saveNpy(data, path):
    np.save(path, data.astype(np.float64))


def excludeFork(df, forkKey="Fork"):
    df_fork_mask = df[forkKey] == True
    return df[~df_fork_mask], df_fork_mask


def readCSV(path, chunksize=None, usecols=None, dtype=None):
    if path in PATH_WITH_TYPES:
        if dtype is None:
            dtype = PATH_WITH_TYPES[path]
        if usecols is None:
            usecols = DEFAULT_USED_COLS[path]

    return pandas.read_csv(path, dtype=dtype, chunksize=chunksize, usecols=usecols)


def saveCSV(dataframe, path, index=False):
    dataframe.to_csv(path, index=index)


def matchingWordRECompiler(word, caseSensitive=False):
    if caseSensitive:
        return re.compile(r'(?<![^\W_]){}(?![^\W_])'.format(word))
    else:
        return re.compile(r'(?<![^\W_]){}(?![^\W_])'.format(word), re.I)

ML_KEYWORDS = [
    "machine[-\s,_]?learning",
    "deep[-\s,_]?learning",
    "statistical[-\s,_]?learning",
    "neural[-\s,_]?network",
    "supervised[-\s,_]?learning",
    "unsupervised[-\s,_]?learning",
    "reinforcement[-\s,_]?learning",
    "artificial[-\s,_]?intelligence",
]

ML_KEYWORDS_RX = [matchingWordRECompiler(w) for w in ML_KEYWORDS]


def matchMLKeyWords(string):
    if not isinstance(string, str):
        return -2
    for idx, rx in enumerate(ML_KEYWORDS_RX):
        if rx.search(string) is not None:
            return idx
    return -1


def mannUandCliffdelta(dist1, dist2):
    d, size = cliffdelta.cliffsDelta(dist1, dist2)
    print(f"Cliff's delta: {size}, d={d}")
    u, p = stats.mannwhitneyu(dist1, dist2, alternative="two-sided")
    print(f"Mann-Whitney-U-test: u={u} p={p}")
    return u, p, d, size


def printValueCountsPercentage(df, name=None, denominator=None, topn=None):
    if name is None:
        cc = df.value_counts(dropna=False)
    else:
        cc = df[name].value_counts(dropna=False)
    if denominator is None:
        denominator = len(df)
    for idx, (n, v) in enumerate(dict(cc).items()):
        if topn is not None and idx >= topn:
            break
        printPercentage(v, denominator, n)


def printPercentage(value, total, prompt=""):
    print(f"{value} / {total} = {value / total * 100}% {prompt}")


def boxplotTwoValues(aList, bList, title="", xticks=["", ""], ylabel="", textFmt="{:.1f}"):
    fig1, ax1 = plt.subplots()
    ax1.boxplot([aList, bList], widths=0.7, showfliers=False, medianprops=dict(color="black", linewidth="3.0"))
    plt.xticks([1, 2], xticks)
    ax1.set_ylabel(ylabel, fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=13)
    ax1.set_title(title, fontsize=25)
    offset_y = (np.median(aList) + np.median(bList)) / 2 * 0.08
    plt.text(0.97, np.median(aList) + offset_y, textFmt.format(np.median(aList)), fontsize=15)
    plt.text(1.97, np.median(bList) + offset_y, textFmt.format(np.median(bList)), fontsize=15)
    plt.show()
    return fig1


def boxplotExp(data, figsize=(8, 6), title="", xtick="", xlabel="", textFmt="{:.1f}"):
    fig1, ax1 = plt.subplots(figsize=figsize)
    bp = ax1.boxplot(data, vert=False, showfliers=False, medianprops=dict(color="black", linewidth="3.0"))
    print("lower_whisker:", bp['whiskers'][0].get_xdata()[1])
    print("upper_whisker:", bp['whiskers'][1].get_xdata()[1])
    print("lower_quartile:", bp['boxes'][0].get_xdata()[1])
    print("upper_quartile:", bp['boxes'][0].get_xdata()[2])
    print("min {}, max {}".format(min(data), max(data)))
    # plt.xscale('symlog')
    # plt.xticks([0,1,10,100], [0,1,10,100])
    ax1.set(ylim=(0.87, 1.13))
    ax1.set_xlabel(xlabel, fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    plt.yticks([], [])
    ax1.set_title(title, fontsize=25)
    plt.text(np.median(data) + 13, 0.995, textFmt.format(np.median(data)), fontsize=15)
    plt.show()
    return fig1


def saveYaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f)


def loadYaml(path):
    with open(path, 'r') as f:
        return yaml.load(f)


def saveJson(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def loadJson(path):
    with open(path, 'r') as f:
        return json.load(f)


def decodeBase64(base64Str):
    return base64.b64decode(base64Str)
    # base64_bytes = base64Str.encode('ascii')
    # message_bytes = base64.b64decode(base64_bytes)
    # decoded_str = message_bytes.decode('utf-8')
    # return decoded_str


def writeToFile(string, path):
    if isinstance(string, bytes):
        fmt = "wb"
    else:
        fmt = 'w'
    with open(path, fmt) as f:
        f.write(string)


def readFile(path) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def timeSeriesToDays(series):
    return secondsToDays(series.total_seconds())


def secondsToDays(seconds):
    return seconds / 86400


def calMADThreshold(data, scale=2.0):
    data = np.array(data)
    med = np.median(data)
    left_mad = np.median(np.abs(data[data < med] - med))
    right_mad = np.median(np.abs(data[data > med] - med))
    if np.isnan(left_mad) or np.isnan(right_mad):
        mad = stats.median_absolute_deviation(data)
        if np.isnan(left_mad):
            left_mad = mad
        if np.isnan(right_mad):
            right_mad = mad
    return med - scale * left_mad, med + scale * right_mad
