[](#downloading-files)Downloading files
=======================================

[](#download-a-single-file)Download a single file
-------------------------------------------------

### [](#huggingface_hub.hf_hub_download)hf\_hub\_download

#### huggingface\_hub.hf\_hub\_download

[](#huggingface_hub.hf_hub_download)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/file_download.py#L663)

( repo\_id: strfilename: strsubfolder: typing.Optional\[str\] = Nonerepo\_type: typing.Optional\[str\] = Nonerevision: typing.Optional\[str\] = Nonelibrary\_name: typing.Optional\[str\] = Nonelibrary\_version: typing.Optional\[str\] = Nonecache\_dir: typing.Union\[str, pathlib.Path, NoneType\] = Nonelocal\_dir: typing.Union\[str, pathlib.Path, NoneType\] = Noneuser\_agent: typing.Union\[typing.Dict, str, NoneType\] = Noneforce\_download: bool = Falseproxies: typing.Optional\[typing.Dict\] = Noneetag\_timeout: float = 10token: typing.Union\[bool, str, NoneType\] = Nonelocal\_files\_only: bool = Falseheaders: typing.Optional\[typing.Dict\[str, str\]\] = Noneendpoint: typing.Optional\[str\] = Noneresume\_download: typing.Optional\[bool\] = Noneforce\_filename: typing.Optional\[str\] = Nonelocal\_dir\_use\_symlinks: typing.Union\[bool, typing.Literal\['auto'\]\] = 'auto' ) → export const metadata = 'undefined';`str`

Expand 16 parameters

Parameters

*   [](#huggingface_hub.hf_hub_download.repo_id)**repo\_id** (`str`) — A user or an organization name and a repo name separated by a `/`.
*   [](#huggingface_hub.hf_hub_download.filename)**filename** (`str`) — The name of the file in the repo.
*   [](#huggingface_hub.hf_hub_download.subfolder)**subfolder** (`str`, _optional_) — An optional value corresponding to a folder inside the model repo.
*   [](#huggingface_hub.hf_hub_download.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if downloading from a dataset or space, `None` or `"model"` if downloading from a model. Default is `None`.
*   [](#huggingface_hub.hf_hub_download.revision)**revision** (`str`, _optional_) — An optional Git revision id which can be a branch name, a tag, or a commit hash.
*   [](#huggingface_hub.hf_hub_download.library_name)**library\_name** (`str`, _optional_) — The name of the library to which the object corresponds.
*   [](#huggingface_hub.hf_hub_download.library_version)**library\_version** (`str`, _optional_) — The version of the library.
*   [](#huggingface_hub.hf_hub_download.cache_dir)**cache\_dir** (`str`, `Path`, _optional_) — Path to the folder where cached files are stored.
*   [](#huggingface_hub.hf_hub_download.local_dir)**local\_dir** (`str` or `Path`, _optional_) — If provided, the downloaded file will be placed under this directory.
*   [](#huggingface_hub.hf_hub_download.user_agent)**user\_agent** (`dict`, `str`, _optional_) — The user-agent info in the form of a dictionary or a string.
*   [](#huggingface_hub.hf_hub_download.force_download)**force\_download** (`bool`, _optional_, defaults to `False`) — Whether the file should be downloaded even if it already exists in the local cache.
*   [](#huggingface_hub.hf_hub_download.proxies)**proxies** (`dict`, _optional_) — Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
*   [](#huggingface_hub.hf_hub_download.etag_timeout)**etag\_timeout** (`float`, _optional_, defaults to `10`) — When fetching ETag, how many seconds to wait for the server to send data before giving up which is passed to `requests.request`.
*   [](#huggingface_hub.hf_hub_download.token)**token** (`str`, `bool`, _optional_) — A token to be used for the download.
    
    *   If `True`, the token is read from the HuggingFace config folder.
    *   If a string, it’s used as the authentication token.
    
*   [](#huggingface_hub.hf_hub_download.local_files_only)**local\_files\_only** (`bool`, _optional_, defaults to `False`) — If `True`, avoid downloading the file and return the path to the local cached file if it exists.
*   [](#huggingface_hub.hf_hub_download.headers)**headers** (`dict`, _optional_) — Additional headers to be sent with the request.

Returns

export const metadata = 'undefined';

`str`

export const metadata = 'undefined';

Local path of file or if networking is off, last version of file cached on disk.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) or [EntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError) or [LocalEntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.LocalEntryNotFoundError) or `EnvironmentError` or `OSError` or `ValueError`

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If the revision to download from cannot be found.
*   [EntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError) — If the file to download cannot be found.
*   [LocalEntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.LocalEntryNotFoundError) — If network is disabled or unavailable and file is not found in cache.
*   [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError) — If `token=True` but the token cannot be found.
*   [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) — If ETag cannot be determined.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If some parameter value is invalid.

Download a given file if it’s not already present in the local cache.

The new cache file layout looks like this:

*   The cache directory contains one subfolder per repo\_id (namespaced by repo type)
*   inside each repo folder:
    *   refs is a list of the latest known revision => commit\_hash pairs
    *   blobs contains the actual file blobs (identified by their git-sha or sha256, depending on whether they’re LFS files or not)
    *   snapshots contains one subfolder per commit, each “commit” contains the subset of the files that have been resolved at that particular commit. Each filename is a symlink to the blob at that particular commit.

[](#huggingface_hub.hf_hub_download.example)

Copied

\[  96\]  .
└── \[ 160\]  models\--julien-c--EsperBERTo-small
    ├── \[ 160\]  blobs
    │   ├── \[321M\]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │   ├── \[ 398\]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
    │   └── \[1.4K\]  d7edf6bd2a681fb0175f7735299831ee1b22b812
    ├── \[  96\]  refs
    │   └── \[  40\]  main
    └── \[ 128\]  snapshots
        ├── \[ 128\]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
        │   ├── \[  52\]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
        │   └── \[  76\]  pytorch\_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        └── \[ 128\]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
            ├── \[  52\]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
            └── \[  76\]  pytorch\_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd

If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this option, the `cache_dir` will not be used and a `.cache/huggingface/` folder will be created at the root of `local_dir` to store some metadata related to the downloaded files. While this mechanism is not as robust as the main cache-system, it’s optimized for regularly pulling the latest version of a repository.

### [](#huggingface_hub.hf_hub_url)hf\_hub\_url

#### huggingface\_hub.hf\_hub\_url

[](#huggingface_hub.hf_hub_url)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/file_download.py#L171)

( repo\_id: strfilename: strsubfolder: typing.Optional\[str\] = Nonerepo\_type: typing.Optional\[str\] = Nonerevision: typing.Optional\[str\] = Noneendpoint: typing.Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.hf_hub_url.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) name and a repo name separated by a `/`.
*   [](#huggingface_hub.hf_hub_url.filename)**filename** (`str`) — The name of the file in the repo.
*   [](#huggingface_hub.hf_hub_url.subfolder)**subfolder** (`str`, _optional_) — An optional value corresponding to a folder inside the repo.
*   [](#huggingface_hub.hf_hub_url.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if downloading from a dataset or space, `None` or `"model"` if downloading from a model. Default is `None`.
*   [](#huggingface_hub.hf_hub_url.revision)**revision** (`str`, _optional_) — An optional Git revision id which can be a branch name, a tag, or a commit hash.

Construct the URL of a file from the given information.

The resolved address can either be a huggingface.co-hosted url, or a link to Cloudfront (a Content Delivery Network, or CDN) for large files which are more than a few MBs.

[](#huggingface_hub.hf_hub_url.example)

Example:

Copied

\>>> from huggingface\_hub import hf\_hub\_url

\>>> hf\_hub\_url(
...     repo\_id="julien-c/EsperBERTo-small", filename="pytorch\_model.bin"
... )
'https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch\_model.bin'

Notes:

Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our bandwidth costs).

Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here because we implement a git-based versioning system on huggingface.co, which means that we store the files on S3/Cloudfront in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache can’t ever be stale.

In terms of client-side caching from this library, we base our caching on the objects’ entity tag (`ETag`), which is an identifier of a specific version of a resource \[1\]\_. An object’s ETag is: its git-sha1 if stored in git, or its sha256 if stored in git-lfs.

References:

*   \[1\] [https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag)

[](#huggingface_hub.snapshot_download)Download a snapshot of the repo
---------------------------------------------------------------------

#### huggingface\_hub.snapshot\_download

[](#huggingface_hub.snapshot_download)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_snapshot_download.py#L20)

( repo\_id: strrepo\_type: typing.Optional\[str\] = Nonerevision: typing.Optional\[str\] = Nonecache\_dir: typing.Union\[str, pathlib.Path, NoneType\] = Nonelocal\_dir: typing.Union\[str, pathlib.Path, NoneType\] = Nonelibrary\_name: typing.Optional\[str\] = Nonelibrary\_version: typing.Optional\[str\] = Noneuser\_agent: typing.Union\[typing.Dict, str, NoneType\] = Noneproxies: typing.Optional\[typing.Dict\] = Noneetag\_timeout: float = 10force\_download: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonelocal\_files\_only: bool = Falseallow\_patterns: typing.Union\[typing.List\[str\], str, NoneType\] = Noneignore\_patterns: typing.Union\[typing.List\[str\], str, NoneType\] = Nonemax\_workers: int = 8tqdm\_class: typing.Optional\[tqdm.asyncio.tqdm\_asyncio\] = Noneheaders: typing.Optional\[typing.Dict\[str, str\]\] = Noneendpoint: typing.Optional\[str\] = Nonelocal\_dir\_use\_symlinks: typing.Union\[bool, typing.Literal\['auto'\]\] = 'auto'resume\_download: typing.Optional\[bool\] = None ) → export const metadata = 'undefined';`str`

Expand 18 parameters

Parameters

*   [](#huggingface_hub.snapshot_download.repo_id)**repo\_id** (`str`) — A user or an organization name and a repo name separated by a `/`.
*   [](#huggingface_hub.snapshot_download.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if downloading from a dataset or space, `None` or `"model"` if downloading from a model. Default is `None`.
*   [](#huggingface_hub.snapshot_download.revision)**revision** (`str`, _optional_) — An optional Git revision id which can be a branch name, a tag, or a commit hash.
*   [](#huggingface_hub.snapshot_download.cache_dir)**cache\_dir** (`str`, `Path`, _optional_) — Path to the folder where cached files are stored.
*   [](#huggingface_hub.snapshot_download.local_dir)**local\_dir** (`str` or `Path`, _optional_) — If provided, the downloaded files will be placed under this directory.
*   [](#huggingface_hub.snapshot_download.library_name)**library\_name** (`str`, _optional_) — The name of the library to which the object corresponds.
*   [](#huggingface_hub.snapshot_download.library_version)**library\_version** (`str`, _optional_) — The version of the library.
*   [](#huggingface_hub.snapshot_download.user_agent)**user\_agent** (`str`, `dict`, _optional_) — The user-agent info in the form of a dictionary or a string.
*   [](#huggingface_hub.snapshot_download.proxies)**proxies** (`dict`, _optional_) — Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
*   [](#huggingface_hub.snapshot_download.etag_timeout)**etag\_timeout** (`float`, _optional_, defaults to `10`) — When fetching ETag, how many seconds to wait for the server to send data before giving up which is passed to `requests.request`.
*   [](#huggingface_hub.snapshot_download.force_download)**force\_download** (`bool`, _optional_, defaults to `False`) — Whether the file should be downloaded even if it already exists in the local cache.
*   [](#huggingface_hub.snapshot_download.token)**token** (`str`, `bool`, _optional_) — A token to be used for the download.
    
    *   If `True`, the token is read from the HuggingFace config folder.
    *   If a string, it’s used as the authentication token.
    
*   [](#huggingface_hub.snapshot_download.headers)**headers** (`dict`, _optional_) — Additional headers to include in the request. Those headers take precedence over the others.
*   [](#huggingface_hub.snapshot_download.local_files_only)**local\_files\_only** (`bool`, _optional_, defaults to `False`) — If `True`, avoid downloading the file and return the path to the local cached file if it exists.
*   [](#huggingface_hub.snapshot_download.allow_patterns)**allow\_patterns** (`List[str]` or `str`, _optional_) — If provided, only files matching at least one pattern are downloaded.
*   [](#huggingface_hub.snapshot_download.ignore_patterns)**ignore\_patterns** (`List[str]` or `str`, _optional_) — If provided, files matching any of the patterns are not downloaded.
*   [](#huggingface_hub.snapshot_download.max_workers)**max\_workers** (`int`, _optional_) — Number of concurrent threads to download files (1 thread = 1 file download). Defaults to 8.
*   [](#huggingface_hub.snapshot_download.tqdm_class)**tqdm\_class** (`tqdm`, _optional_) — If provided, overwrites the default behavior for the progress bar. Passed argument must inherit from `tqdm.auto.tqdm` or at least mimic its behavior. Note that the `tqdm_class` is not passed to each individual download. Defaults to the custom HF progress bar that can be disabled by setting `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.

Returns

export const metadata = 'undefined';

`str`

export const metadata = 'undefined';

folder path of the repo snapshot.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) or `EnvironmentError` or `OSError` or `ValueError`

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If the revision to download from cannot be found.
*   [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError) — If `token=True` and the token cannot be found.
*   [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) — if ETag cannot be determined.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — if some parameter value is invalid.

Download repo files.

Download a whole snapshot of a repo’s files at the specified revision. This is useful when you want all files from a repo, because you don’t know which ones you will need a priori. All files are nested inside a folder in order to keep their actual filename relative to that folder. You can also filter which files to download using `allow_patterns` and `ignore_patterns`.

If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this option, the `cache_dir` will not be used and a `.cache/huggingface/` folder will be created at the root of `local_dir` to store some metadata related to the downloaded files. While this mechanism is not as robust as the main cache-system, it’s optimized for regularly pulling the latest version of a repository.

An alternative would be to clone the repo but this requires git and git-lfs to be installed and properly configured. It is also not possible to filter which files to download when cloning a repository using git.

[](#get-metadata-about-a-file)Get metadata about a file
-------------------------------------------------------

### [](#huggingface_hub.get_hf_file_metadata)get\_hf\_file\_metadata

#### huggingface\_hub.get\_hf\_file\_metadata

[](#huggingface_hub.get_hf_file_metadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/file_download.py#L1246)

( url: strtoken: typing.Union\[bool, str, NoneType\] = Noneproxies: typing.Optional\[typing.Dict\] = Nonetimeout: typing.Optional\[float\] = 10library\_name: typing.Optional\[str\] = Nonelibrary\_version: typing.Optional\[str\] = Noneuser\_agent: typing.Union\[typing.Dict, str, NoneType\] = Noneheaders: typing.Optional\[typing.Dict\[str, str\]\] = None )

Parameters

*   [](#huggingface_hub.get_hf_file_metadata.url)**url** (`str`) — File url, for example returned by [hf\_hub\_url()](/docs/huggingface_hub/v0.29.2/en/package_reference/file_download#huggingface_hub.hf_hub_url).
*   [](#huggingface_hub.get_hf_file_metadata.token)**token** (`str` or `bool`, _optional_) — A token to be used for the download.
    
    *   If `True`, the token is read from the HuggingFace config folder.
    *   If `False` or `None`, no token is provided.
    *   If a string, it’s used as the authentication token.
    
*   [](#huggingface_hub.get_hf_file_metadata.proxies)**proxies** (`dict`, _optional_) — Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
*   [](#huggingface_hub.get_hf_file_metadata.timeout)**timeout** (`float`, _optional_, defaults to 10) — How many seconds to wait for the server to send metadata before giving up.
*   [](#huggingface_hub.get_hf_file_metadata.library_name)**library\_name** (`str`, _optional_) — The name of the library to which the object corresponds.
*   [](#huggingface_hub.get_hf_file_metadata.library_version)**library\_version** (`str`, _optional_) — The version of the library.
*   [](#huggingface_hub.get_hf_file_metadata.user_agent)**user\_agent** (`dict`, `str`, _optional_) — The user-agent info in the form of a dictionary or a string.
*   [](#huggingface_hub.get_hf_file_metadata.headers)**headers** (`dict`, _optional_) — Additional headers to be sent with the request.

Fetch metadata of a file versioned on the Hub for a given url.

### [](#huggingface_hub.HfFileMetadata)HfFileMetadata

### class huggingface\_hub.HfFileMetadata

[](#huggingface_hub.HfFileMetadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/file_download.py#L147)

( commit\_hash: typing.Optional\[str\]etag: typing.Optional\[str\]location: strsize: typing.Optional\[int\] )

Parameters

*   [](#huggingface_hub.HfFileMetadata.commit_hash)**commit\_hash** (`str`, _optional_) — The commit\_hash related to the file.
*   [](#huggingface_hub.HfFileMetadata.etag)**etag** (`str`, _optional_) — Etag of the file on the server.
*   [](#huggingface_hub.HfFileMetadata.location)**location** (`str`) — Location where to download the file. Can be a Hub url or not (CDN).
*   [](#huggingface_hub.HfFileMetadata.size)**size** (`size`) — Size of the file. In case of an LFS file, contains the size of the actual LFS file, not the pointer.

Data structure containing information about a file versioned on the Hub.

Returned by [get\_hf\_file\_metadata()](/docs/huggingface_hub/v0.29.2/en/package_reference/file_download#huggingface_hub.get_hf_file_metadata) based on a URL.

[](#caching)Caching
-------------------

The methods displayed above are designed to work with a caching system that prevents re-downloading files. The caching system was updated in v0.8.0 to become the central cache-system shared across libraries that depend on the Hub.

Read the [cache-system guide](../guides/manage-cache) for a detailed presentation of caching at at HF.

[< \> Update on GitHub](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/package_reference/file_download.md)

HfApi Client

[←Hugging Face Hub API](/docs/huggingface_hub/en/package_reference/hf_api) [Mixins & serialization methods→](/docs/huggingface_hub/en/package_reference/mixins)