[](#hfapi-client)HfApi Client
=============================

Below is the documentation for the `HfApi` class, which serves as a Python wrapper for the Hugging Face Hub’s API.

All methods from the `HfApi` are also accessible from the package’s root directly. Both approaches are detailed below.

Using the root method is more straightforward but the [HfApi](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi) class gives you more flexibility. In particular, you can pass a token that will be reused in all HTTP calls. This is different than `huggingface-cli login` or [login()](/docs/huggingface_hub/v0.29.2/en/package_reference/authentication#huggingface_hub.login) as the token is not persisted on the machine. It is also possible to provide a different endpoint or configure a custom user-agent.

Copied

from huggingface\_hub import HfApi, list\_models

\# Use root method
models = list\_models()

\# Or configure a HfApi client
hf\_api = HfApi(
    endpoint="https://huggingface.co", \# Can be a Private Hub endpoint.
    token="hf\_xxx", \# Token is not persisted on the machine.
)
models = hf\_api.list\_models()

[](#huggingface_hub.HfApi)HfApi
-------------------------------

### class huggingface\_hub.HfApi

[](#huggingface_hub.HfApi)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1557)

( endpoint: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = Nonelibrary\_name: Optional\[str\] = Nonelibrary\_version: Optional\[str\] = Noneuser\_agent: Union\[Dict, str, None\] = Noneheaders: Optional\[Dict\[str, str\]\] = None )

Parameters

*   [](#huggingface_hub.HfApi.endpoint)**endpoint** (`str`, _optional_) — Endpoint of the Hub. Defaults to [https://huggingface.co](https://huggingface.co).
*   [](#huggingface_hub.HfApi.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.library_name)**library\_name** (`str`, _optional_) — The name of the library that is making the HTTP request. Will be added to the user-agent header. Example: `"transformers"`.
*   [](#huggingface_hub.HfApi.library_version)**library\_version** (`str`, _optional_) — The version of the library that is making the HTTP request. Will be added to the user-agent header. Example: `"4.24.0"`.
*   [](#huggingface_hub.HfApi.user_agent)**user\_agent** (`str`, `dict`, _optional_) — The user agent info in the form of a dictionary or a single string. It will be completed with information about the installed packages.
*   [](#huggingface_hub.HfApi.headers)**headers** (`dict`, _optional_) — Additional headers to be sent with each request. Example: `{"X-My-Header": "value"}`. Headers passed here are taking precedence over the default headers.

Client to interact with the Hugging Face Hub via HTTP.

The client is initialized with some high-level settings used in all requests made to the Hub (HF endpoint, authentication, user agents…). Using the `HfApi` client is preferred but not mandatory as all of its public methods are exposed directly at the root of `huggingface_hub`.

#### accept\_access\_request

[](#huggingface_hub.HfApi.accept_access_request)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8524)

( repo\_id: struser: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Expand 4 parameters

Parameters

*   [](#huggingface_hub.HfApi.accept_access_request.repo_id)**repo\_id** (`str`) — The id of the repo to accept access request for.
*   [](#huggingface_hub.HfApi.accept_access_request.user)**user** (`str`) — The username of the user which access request should be accepted.
*   [](#huggingface_hub.HfApi.accept_access_request.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to accept access request for. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.accept_access_request.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user does not exist on the Hub.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user access request cannot be found.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user access request is already in the accepted list.

Accept an access request from a user for a given gated repo.

Once the request is accepted, the user will be able to download any file of the repo and access the community tab. If the approval mode is automatic, you don’t have to accept requests manually. An accepted request can be cancelled or rejected at any time using [cancel\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request) and [reject\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request).

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

#### add\_collection\_item

[](#huggingface_hub.HfApi.add_collection_item)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8076)

( collection\_slug: stritem\_id: stritem\_type: CollectionItemType\_Tnote: Optional\[str\] = Noneexists\_ok: bool = Falsetoken: Union\[bool, str, None\] = None )

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.add_collection_item.collection_slug)**collection\_slug** (`str`) — Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
*   [](#huggingface_hub.HfApi.add_collection_item.item_id)**item\_id** (`str`) — ID of the item to add to the collection. It can be the ID of a repo on the Hub (e.g. `"facebook/bart-large-mnli"`) or a paper id (e.g. `"2307.09288"`).
*   [](#huggingface_hub.HfApi.add_collection_item.item_type)**item\_type** (`str`) — Type of the item to add. Can be one of `"model"`, `"dataset"`, `"space"` or `"paper"`.
*   [](#huggingface_hub.HfApi.add_collection_item.note)**note** (`str`, _optional_) — A note to attach to the item in the collection. The maximum size for a note is 500 characters.
*   [](#huggingface_hub.HfApi.add_collection_item.exists_ok)**exists\_ok** (`bool`, _optional_) — If `True`, do not raise an error if item already exists.
*   [](#huggingface_hub.HfApi.add_collection_item.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the item you try to add to the collection does not exist on the Hub.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 409 if the item you try to add to the collection is already in the collection (and exists\_ok=False)

Add an item to a collection on the Hub.

Returns: [Collection](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.Collection)

[](#huggingface_hub.HfApi.add_collection_item.example)

Example:

Copied

\>>> from huggingface\_hub import add\_collection\_item
\>>> collection = add\_collection\_item(
...     collection\_slug="davanstrien/climate-64f99dc2a5067f6b65531bab",
...     item\_id="pierre-loic/climate-news-articles",
...     item\_type="dataset"
... )
\>>> collection.items\[-1\].item\_id
"pierre-loic/climate-news-articles"
\# ^item got added to the collection on last position

\# Add item with a note
\>>> add\_collection\_item(
...     collection\_slug="davanstrien/climate-64f99dc2a5067f6b65531bab",
...     item\_id="datasets/climate\_fever",
...     item\_type="dataset"
...     note="This dataset adopts the FEVER methodology that consists of 1,535 real-world claims regarding climate-change collected on the internet."
... )
(...)

#### add\_space\_secret

[](#huggingface_hub.HfApi.add_space_secret)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6705)

( repo\_id: strkey: strvalue: strdescription: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.add_space_secret.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.add_space_secret.key)**key** (`str`) — Secret key. Example: `"GITHUB_API_KEY"`
*   [](#huggingface_hub.HfApi.add_space_secret.value)**value** (`str`) — Secret value. Example: `"your_github_api_key"`.
*   [](#huggingface_hub.HfApi.add_space_secret.description)**description** (`str`, _optional_) — Secret description. Example: `"Github API key to access the Github API"`.
*   [](#huggingface_hub.HfApi.add_space_secret.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Adds or updates a secret in a Space.

Secrets allow to set secret keys or tokens to a Space without hardcoding them. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

#### add\_space\_variable

[](#huggingface_hub.HfApi.add_space_variable)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6793)

( repo\_id: strkey: strvalue: strdescription: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.add_space_variable.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.add_space_variable.key)**key** (`str`) — Variable key. Example: `"MODEL_REPO_ID"`
*   [](#huggingface_hub.HfApi.add_space_variable.value)**value** (`str`) — Variable value. Example: `"the_model_repo_id"`.
*   [](#huggingface_hub.HfApi.add_space_variable.description)**description** (`str`) — Description of the variable. Example: `"Model Repo ID of the implemented model"`.
*   [](#huggingface_hub.HfApi.add_space_variable.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Adds or updates a variable in a Space.

Variables allow to set environment variables to a Space without hardcoding them. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables)

#### auth\_check

[](#huggingface_hub.HfApi.auth_check)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9432)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.auth_check.repo_id)**repo\_id** (`str`) — The repository to check for access. Format should be `"user/repo_name"`. Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.auth_check.repo_type)**repo\_type** (`str`, _optional_) — The type of the repository. Should be one of `"model"`, `"dataset"`, or `"space"`. If not specified, the default is `"model"`.
*   [](#huggingface_hub.HfApi.auth_check.token)**token** `(Union[bool, str, None]`, _optional_) — A valid user access token. If not provided, the locally saved token will be used, which is the recommended authentication method. Set to `False` to disable authentication. Refer to: [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [GatedRepoError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.GatedRepoError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — Raised if the repository does not exist, is private, or the user does not have access. This can occur if the `repo_id` or `repo_type` is incorrect or if the repository is private but the user is not authenticated.
    
*   [GatedRepoError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.GatedRepoError) — Raised if the repository exists but is gated and the user is not authorized to access it.
    

Check if the provided user token has access to a specific repository on the Hugging Face Hub.

This method verifies whether the user, authenticated via the provided token, has access to the specified repository. If the repository is not found or if the user lacks the required permissions to access it, the method raises an appropriate exception.

Example:

[](#huggingface_hub.HfApi.auth_check.example)

Check if the user has access to a repository:

Copied

\>>> from huggingface\_hub import auth\_check
\>>> from huggingface\_hub.utils import GatedRepoError, RepositoryNotFoundError

try:
    auth\_check("user/my-cool-model")
except GatedRepoError:
    \# Handle gated repository error
    print("You do not have permission to access this gated repository.")
except RepositoryNotFoundError:
    \# Handle repository not found error
    print("The repository was not found or you do not have access.")

In this example:

*   If the user has access, the method completes successfully.
*   If the repository is gated or does not exist, appropriate exceptions are raised, allowing the user to handle them accordingly.

#### cancel\_access\_request

[](#huggingface_hub.HfApi.cancel_access_request)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8484)

( repo\_id: struser: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Expand 4 parameters

Parameters

*   [](#huggingface_hub.HfApi.cancel_access_request.repo_id)**repo\_id** (`str`) — The id of the repo to cancel access request for.
*   [](#huggingface_hub.HfApi.cancel_access_request.user)**user** (`str`) — The username of the user which access request should be cancelled.
*   [](#huggingface_hub.HfApi.cancel_access_request.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to cancel access request for. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.cancel_access_request.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user does not exist on the Hub.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user access request cannot be found.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user access request is already in the pending list.

Cancel an access request from a user for a given gated repo.

A cancelled request will go back to the pending list and the user will lose access to the repo.

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

#### change\_discussion\_status

[](#huggingface_hub.HfApi.change_discussion_status)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6447)

( repo\_id: strdiscussion\_num: intnew\_status: Literal\['open', 'closed'\]token: Union\[bool, str, None\] = Nonecomment: Optional\[str\] = Nonerepo\_type: Optional\[str\] = None ) → export const metadata = 'undefined';[DiscussionStatusChange](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionStatusChange)

Parameters

*   [](#huggingface_hub.HfApi.change_discussion_status.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.change_discussion_status.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.change_discussion_status.new_status)**new\_status** (`str`) — The new status for the discussion, either `"open"` or `"closed"`.
*   [](#huggingface_hub.HfApi.change_discussion_status.comment)**comment** (`str`, _optional_) — An optional comment to post with the status change.
*   [](#huggingface_hub.HfApi.change_discussion_status.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.change_discussion_status.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[DiscussionStatusChange](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionStatusChange)

export const metadata = 'undefined';

the status change event

Closes or re-opens a Discussion or Pull Request.

[](#huggingface_hub.HfApi.change_discussion_status.example)

Examples:

Copied

\>>> new\_title = "New title, fixing a typo"
\>>> HfApi().rename\_discussion(
...     repo\_id="username/repo\_name",
...     discussion\_num=34
...     new\_title=new\_title
... )
\# DiscussionStatusChange(id='deadbeef0000000', type='status-change', ...)

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### comment\_discussion

[](#huggingface_hub.HfApi.comment_discussion)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6298)

( repo\_id: strdiscussion\_num: intcomment: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None ) → export const metadata = 'undefined';[DiscussionComment](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionComment)

Parameters

*   [](#huggingface_hub.HfApi.comment_discussion.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.comment_discussion.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.comment_discussion.comment)**comment** (`str`) — The content of the comment to create. Comments support markdown formatting.
*   [](#huggingface_hub.HfApi.comment_discussion.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.comment_discussion.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[DiscussionComment](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionComment)

export const metadata = 'undefined';

the newly created comment

Creates a new comment on the given Discussion.

[](#huggingface_hub.HfApi.comment_discussion.example)

Examples:

Copied

\>>> comment = """
... Hello @otheruser!
...
... \# This is a title
...
... \*\*This is bold\*\*, \*this is italic\* and ~this is strikethrough~
... And \[this\](http://url) is a link
... """

\>>> HfApi().comment\_discussion(
...     repo\_id="username/repo\_name",
...     discussion\_num=34
...     comment=comment
... )
\# DiscussionComment(id='deadbeef0000000', type='comment', ...)

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### create\_branch

[](#huggingface_hub.HfApi.create_branch)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5639)

( repo\_id: strbranch: strrevision: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = Noneexist\_ok: bool = False )

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.create_branch.repo_id)**repo\_id** (`str`) — The repository in which the branch will be created. Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.create_branch.branch)**branch** (`str`) — The name of the branch to create.
*   [](#huggingface_hub.HfApi.create_branch.revision)**revision** (`str`, _optional_) — The git revision to create the branch from. It can be a branch name or the OID/SHA of a commit, as a hexadecimal string. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.create_branch.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.create_branch.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if creating a branch on a dataset or space, `None` or `"model"` if tagging a model. Default is `None`.
*   [](#huggingface_hub.HfApi.create_branch.exist_ok)**exist\_ok** (`bool`, _optional_, defaults to `False`) — If `True`, do not raise an error if branch already exists.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) or [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) — If invalid reference for a branch. Ex: `refs/pr/5` or ‘refs/foo/bar’.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — If the branch already exists on the repo (error 409) and `exist_ok` is set to `False`.

Create a new branch for a repo on the Hub, starting from the specified revision (defaults to `main`). To find a revision suiting your needs, you can use [list\_repo\_refs()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_refs) or [list\_repo\_commits()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_commits).

#### create\_collection

[](#huggingface_hub.HfApi.create_collection)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7899)

( title: strnamespace: Optional\[str\] = Nonedescription: Optional\[str\] = Noneprivate: bool = Falseexists\_ok: bool = Falsetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.create_collection.title)**title** (`str`) — Title of the collection to create. Example: `"Recent models"`.
*   [](#huggingface_hub.HfApi.create_collection.namespace)**namespace** (`str`, _optional_) — Namespace of the collection to create (username or org). Will default to the owner name.
*   [](#huggingface_hub.HfApi.create_collection.description)**description** (`str`, _optional_) — Description of the collection to create.
*   [](#huggingface_hub.HfApi.create_collection.private)**private** (`bool`, _optional_) — Whether the collection should be private or not. Defaults to `False` (i.e. public collection).
*   [](#huggingface_hub.HfApi.create_collection.exists_ok)**exists\_ok** (`bool`, _optional_) — If `True`, do not raise an error if collection already exists.
*   [](#huggingface_hub.HfApi.create_collection.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Create a new Collection on the Hub.

Returns: [Collection](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.Collection)

[](#huggingface_hub.HfApi.create_collection.example)

Example:

Copied

\>>> from huggingface\_hub import create\_collection
\>>> collection = create\_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
\>>> collection.slug
"username/iccv-2023-64f9a55bb3115b4f513ec026"

#### create\_commit

[](#huggingface_hub.HfApi.create_commit)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3812)

( repo\_id: stroperations: Iterable\[CommitOperation\]commit\_message: strcommit\_description: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Nonenum\_threads: int = 5parent\_commit: Optional\[str\] = Nonerun\_as\_future: bool = False ) → export const metadata = 'undefined';[CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) or `Future`

Expand 11 parameters

Parameters

*   [](#huggingface_hub.HfApi.create_commit.repo_id)**repo\_id** (`str`) — The repository in which the commit will be created, for example: `"username/custom_transformers"`
*   [](#huggingface_hub.HfApi.create_commit.operations)**operations** (`Iterable` of `CommitOperation()`) — An iterable of operations to include in the commit, either:
    
    *   [CommitOperationAdd](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd) to upload a file
    *   [CommitOperationDelete](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitOperationDelete) to delete a file
    *   [CommitOperationCopy](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy) to copy a file
    
    Operation objects will be mutated to include information relative to the upload. Do not reuse the same objects for multiple commits.
    
*   [](#huggingface_hub.HfApi.create_commit.commit_message)**commit\_message** (`str`) — The summary (first line) of the commit that will be created.
*   [](#huggingface_hub.HfApi.create_commit.commit_description)**commit\_description** (`str`, _optional_) — The description of the commit that will be created
*   [](#huggingface_hub.HfApi.create_commit.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.create_commit.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.create_commit.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.create_commit.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened against this branch. If `revision` is set and is not a branch name (example: a commit oid), an `RevisionNotFoundError` is returned by the server.
*   [](#huggingface_hub.HfApi.create_commit.num_threads)**num\_threads** (`int`, _optional_) — Number of concurrent threads for uploading files. Defaults to 5. Setting it to 2 means at most 2 files will be uploaded concurrently.
*   [](#huggingface_hub.HfApi.create_commit.parent_commit)**parent\_commit** (`str`, _optional_) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.
*   [](#huggingface_hub.HfApi.create_commit.run_as_future)**run\_as\_future** (`bool`, _optional_) — Whether or not to run this method in the background. Background jobs are run sequentially without blocking the main thread. Passing `run_as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) object. Defaults to `False`.

Returns

export const metadata = 'undefined';

[CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) or `Future`

export const metadata = 'undefined';

Instance of [CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) containing information about the newly created commit (commit hash, commit url, pr url, commit message,…). If `run_as_future=True` is passed, returns a Future object which will contain the result when executed.

Raises

export const metadata = 'undefined';

`ValueError` or [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

export const metadata = 'undefined';

*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If commit message is empty.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If parent commit is not a valid commit OID.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If a README.md file with an invalid metadata section is committed. In this case, the commit will fail early, before trying to upload any file.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If `create_pr` is `True` and revision is neither `None` nor `"main"`.
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.

Creates a commit in the given repo, deleting & uploading files as needed.

The input list of `CommitOperation` will be mutated during the commit process. Do not reuse the same objects for multiple commits.

`create_commit` assumes that the repo already exists on the Hub. If you get a Client error 404, please make sure you are authenticated and that `repo_id` and `repo_type` are set correctly. If repo does not exist, create it first using [create\_repo()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo).

`create_commit` is limited to 25k LFS files and a 1GB payload for regular files.

#### create\_discussion

[](#huggingface_hub.HfApi.create_discussion)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6119)

( repo\_id: strtitle: strtoken: Union\[bool, str, None\] = Nonedescription: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonepull\_request: bool = False )

Parameters

*   [](#huggingface_hub.HfApi.create_discussion.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.create_discussion.title)**title** (`str`) — The title of the discussion. It can be up to 200 characters long, and must be at least 3 characters long. Leading and trailing whitespaces will be stripped.
*   [](#huggingface_hub.HfApi.create_discussion.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.create_discussion.description)**description** (`str`, _optional_) — An optional description for the Pull Request. Defaults to `"Discussion opened with the huggingface_hub Python library"`
*   [](#huggingface_hub.HfApi.create_discussion.pull_request)**pull\_request** (`bool`, _optional_) — Whether to create a Pull Request or discussion. If `True`, creates a Pull Request. If `False`, creates a discussion. Defaults to `False`.
*   [](#huggingface_hub.HfApi.create_discussion.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.

Creates a Discussion or Pull Request.

Pull Requests created programmatically will be in `"draft"` status.

Creating a Pull Request with changes can also be done at once with [HfApi.create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit).

Returns: [DiscussionWithDetails](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionWithDetails)

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### create\_inference\_endpoint

[](#huggingface_hub.HfApi.create_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7322)

( name: strrepository: strframework: straccelerator: strinstance\_size: strinstance\_type: strregion: strvendor: straccount\_id: Optional\[str\] = Nonemin\_replica: int = 0max\_replica: int = 1scale\_to\_zero\_timeout: int = 15revision: Optional\[str\] = Nonetask: Optional\[str\] = Nonecustom\_image: Optional\[Dict\] = Nonesecrets: Optional\[Dict\[str, str\]\] = Nonetype: InferenceEndpointType = <InferenceEndpointType.PROTECTED: 'protected'>namespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Expand 19 parameters

Parameters

*   [](#huggingface_hub.HfApi.create_inference_endpoint.name)**name** (`str`) — The unique name for the new Inference Endpoint.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.repository)**repository** (`str`) — The name of the model repository associated with the Inference Endpoint (e.g. `"gpt2"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.framework)**framework** (`str`) — The machine learning framework used for the model (e.g. `"custom"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.accelerator)**accelerator** (`str`) — The hardware accelerator to be used for inference (e.g. `"cpu"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.instance_size)**instance\_size** (`str`) — The size or type of the instance to be used for hosting the model (e.g. `"x4"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.instance_type)**instance\_type** (`str`) — The cloud instance type where the Inference Endpoint will be deployed (e.g. `"intel-icl"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.region)**region** (`str`) — The cloud region in which the Inference Endpoint will be created (e.g. `"us-east-1"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.vendor)**vendor** (`str`) — The cloud provider or vendor where the Inference Endpoint will be hosted (e.g. `"aws"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.account_id)**account\_id** (`str`, _optional_) — The account ID used to link a VPC to a private Inference Endpoint (if applicable).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.min_replica)**min\_replica** (`int`, _optional_) — The minimum number of replicas (instances) to keep running for the Inference Endpoint. Defaults to 0.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.max_replica)**max\_replica** (`int`, _optional_) — The maximum number of replicas (instances) to scale to for the Inference Endpoint. Defaults to 1.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.scale_to_zero_timeout)**scale\_to\_zero\_timeout** (`int`, _optional_) — The duration in minutes before an inactive endpoint is scaled to zero. Defaults to 15.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.revision)**revision** (`str`, _optional_) — The specific model revision to deploy on the Inference Endpoint (e.g. `"6c0e6080953db56375760c0471a8c5f2929baf11"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.task)**task** (`str`, _optional_) — The task on which to deploy the model (e.g. `"text-classification"`).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.custom_image)**custom\_image** (`Dict`, _optional_) — A custom Docker image to use for the Inference Endpoint. This is useful if you want to deploy an Inference Endpoint running on the `text-generation-inference` (TGI) framework (see examples).
*   [](#huggingface_hub.HfApi.create_inference_endpoint.secrets)**secrets** (`Dict[str, str]`, _optional_) — Secret values to inject in the container environment.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.type)**type** (\[\`InferenceEndpointType\]`, *optional*) -- The type of the Inference Endpoint, which can be` “protected”`(default),`“public”`or`“private”\`.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace where the Inference Endpoint will be created. Defaults to the current user’s namespace.
*   [](#huggingface_hub.HfApi.create_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

information about the updated Inference Endpoint.

Create a new Inference Endpoint.

[](#huggingface_hub.HfApi.create_inference_endpoint.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> endpoint = api.create\_inference\_endpoint(
...     "my-endpoint-name",
...     repository="gpt2",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="cpu",
...     vendor="aws",
...     region="us-east-1",
...     type\="protected",
...     instance\_size="x2",
...     instance\_type="intel-icl",
... )
\>>> endpoint
InferenceEndpoint(name='my-endpoint-name', status="pending",...)

\# Run inference on the endpoint
\>>> endpoint.client.text\_generation(...)
"..."

[](#huggingface_hub.HfApi.create_inference_endpoint.example-2)

Copied

\# Start an Inference Endpoint running Zephyr-7b-beta on TGI
\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> endpoint = api.create\_inference\_endpoint(
...     "aws-zephyr-7b-beta-0486",
...     repository="HuggingFaceH4/zephyr-7b-beta",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="gpu",
...     vendor="aws",
...     region="us-east-1",
...     type\="protected",
...     instance\_size="x1",
...     instance\_type="nvidia-a10g",
...     custom\_image={
...         "health\_route": "/health",
...         "env": {
...             "MAX\_BATCH\_PREFILL\_TOKENS": "2048",
...             "MAX\_INPUT\_LENGTH": "1024",
...             "MAX\_TOTAL\_TOKENS": "1512",
...             "MODEL\_ID": "/repository"
...         },
...         "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
...     },
...    secrets={"MY\_SECRET\_KEY": "secret\_value"},
... )

#### create\_pull\_request

[](#huggingface_hub.HfApi.create_pull_request)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6211)

( repo\_id: strtitle: strtoken: Union\[bool, str, None\] = Nonedescription: Optional\[str\] = Nonerepo\_type: Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.HfApi.create_pull_request.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.create_pull_request.title)**title** (`str`) — The title of the discussion. It can be up to 200 characters long, and must be at least 3 characters long. Leading and trailing whitespaces will be stripped.
*   [](#huggingface_hub.HfApi.create_pull_request.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.create_pull_request.description)**description** (`str`, _optional_) — An optional description for the Pull Request. Defaults to `"Discussion opened with the huggingface_hub Python library"`
*   [](#huggingface_hub.HfApi.create_pull_request.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.

Creates a Pull Request . Pull Requests created programmatically will be in `"draft"` status.

Creating a Pull Request with changes can also be done at once with [HfApi.create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit);

This is a wrapper around [HfApi.create\_discussion()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_discussion).

Returns: [DiscussionWithDetails](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionWithDetails)

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### create\_repo

[](#huggingface_hub.HfApi.create_repo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3382)

( repo\_id: strtoken: Union\[str, bool, None\] = Noneprivate: Optional\[bool\] = Nonerepo\_type: Optional\[str\] = Noneexist\_ok: bool = Falseresource\_group\_id: Optional\[str\] = Nonespace\_sdk: Optional\[str\] = Nonespace\_hardware: Optional\[SpaceHardware\] = Nonespace\_storage: Optional\[SpaceStorage\] = Nonespace\_sleep\_time: Optional\[int\] = Nonespace\_secrets: Optional\[List\[Dict\[str, str\]\]\] = Nonespace\_variables: Optional\[List\[Dict\[str, str\]\]\] = None ) → export const metadata = 'undefined';[RepoUrl](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.RepoUrl)

Expand 12 parameters

Parameters

*   [](#huggingface_hub.HfApi.create_repo.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.create_repo.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.create_repo.private)**private** (`bool`, _optional_) — Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
*   [](#huggingface_hub.HfApi.create_repo.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.create_repo.exist_ok)**exist\_ok** (`bool`, _optional_, defaults to `False`) — If `True`, do not raise an error if repo already exists.
*   [](#huggingface_hub.HfApi.create_repo.resource_group_id)**resource\_group\_id** (`str`, _optional_) — Resource group in which to create the repo. Resource groups is only available for organizations and allow to define which members of the organization can access the resource. The ID of a resource group can be found in the URL of the resource’s page on the Hub (e.g. `"66670e5163145ca562cb1988"`). To learn more about resource groups, see [https://huggingface.co/docs/hub/en/security-resource-groups](https://huggingface.co/docs/hub/en/security-resource-groups).
*   [](#huggingface_hub.HfApi.create_repo.space_sdk)**space\_sdk** (`str`, _optional_) — Choice of SDK to use if repo\_type is “space”. Can be “streamlit”, “gradio”, “docker”, or “static”.
*   [](#huggingface_hub.HfApi.create_repo.space_hardware)**space\_hardware** (`SpaceHardware` or `str`, _optional_) — Choice of Hardware if repo\_type is “space”. See [SpaceHardware](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceHardware) for a complete list.
*   [](#huggingface_hub.HfApi.create_repo.space_storage)**space\_storage** (`SpaceStorage` or `str`, _optional_) — Choice of persistent storage tier. Example: `"small"`. See [SpaceStorage](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceStorage) for a complete list.
*   [](#huggingface_hub.HfApi.create_repo.space_sleep_time)**space\_sleep\_time** (`int`, _optional_) — Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don’t want your Space to sleep (default behavior for upgraded hardware). For free hardware, you can’t configure the sleep time (value is fixed to 48 hours of inactivity). See [https://huggingface.co/docs/hub/spaces-gpus#sleep-time](https://huggingface.co/docs/hub/spaces-gpus#sleep-time) for more details.
*   [](#huggingface_hub.HfApi.create_repo.space_secrets)**space\_secrets** (`List[Dict[str, str]]`, _optional_) — A list of secret keys to set in your Space. Each item is in the form `{"key": ..., "value": ..., "description": ...}` where description is optional. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).
*   [](#huggingface_hub.HfApi.create_repo.space_variables)**space\_variables** (`List[Dict[str, str]]`, _optional_) — A list of public environment variables to set in your Space. Each item is in the form `{"key": ..., "value": ..., "description": ...}` where description is optional. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables).

Returns

export const metadata = 'undefined';

[RepoUrl](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.RepoUrl)

export const metadata = 'undefined';

URL to the newly created repo. Value is a subclass of `str` containing attributes like `endpoint`, `repo_type` and `repo_id`.

Create an empty repo on the HuggingFace Hub.

#### create\_tag

[](#huggingface_hub.HfApi.create_tag)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5771)

( repo\_id: strtag: strtag\_message: Optional\[str\] = Nonerevision: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = Noneexist\_ok: bool = False )

Expand 7 parameters

Parameters

*   [](#huggingface_hub.HfApi.create_tag.repo_id)**repo\_id** (`str`) — The repository in which a commit will be tagged. Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.create_tag.tag)**tag** (`str`) — The name of the tag to create.
*   [](#huggingface_hub.HfApi.create_tag.tag_message)**tag\_message** (`str`, _optional_) — The description of the tag to create.
*   [](#huggingface_hub.HfApi.create_tag.revision)**revision** (`str`, _optional_) — The git revision to tag. It can be a branch name or the OID/SHA of a commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.create_tag.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.create_tag.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if tagging a dataset or space, `None` or `"model"` if tagging a model. Default is `None`.
*   [](#huggingface_hub.HfApi.create_tag.exist_ok)**exist\_ok** (`bool`, _optional_, defaults to `False`) — If `True`, do not raise an error if tag already exists.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) or [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If revision is not found (error 404) on the repo.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — If the branch already exists on the repo (error 409) and `exist_ok` is set to `False`.

Tag a given commit of a repo on the Hub.

#### create\_webhook

[](#huggingface_hub.HfApi.create_webhook)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8803)

( url: strwatched: List\[Union\[Dict, WebhookWatchedItem\]\]domains: Optional\[List\[constants.WEBHOOK\_DOMAIN\_T\]\] = Nonesecret: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

Parameters

*   [](#huggingface_hub.HfApi.create_webhook.url)**url** (`str`) — URL to send the payload to.
*   [](#huggingface_hub.HfApi.create_webhook.watched)**watched** (`List[WebhookWatchedItem]`) — List of [WebhookWatchedItem](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookWatchedItem) to be watched by the webhook. It can be users, orgs, models, datasets or spaces. Watched items can also be provided as plain dictionaries.
*   [](#huggingface_hub.HfApi.create_webhook.domains)**domains** (`List[Literal["repo", "discussion"]]`, optional) — List of domains to watch. It can be “repo”, “discussion” or both.
*   [](#huggingface_hub.HfApi.create_webhook.secret)**secret** (`str`, optional) — A secret to sign the payload with.
*   [](#huggingface_hub.HfApi.create_webhook.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

export const metadata = 'undefined';

Info about the newly created webhook.

Create a new webhook.

[](#huggingface_hub.HfApi.create_webhook.example)

Example:

Copied

\>>> from huggingface\_hub import create\_webhook
\>>> payload = create\_webhook(
...     watched=\[{"type": "user", "name": "julien-c"}, {"type": "org", "name": "HuggingFaceH4"}\],
...     url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
...     domains=\["repo", "discussion"\],
...     secret="my-secret",
... )
\>>> print(payload)
WebhookInfo(
    id\="654bbbc16f2ec14d77f109cc",
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    watched=\[WebhookWatchedItem(type\="user", name="julien-c"), WebhookWatchedItem(type\="org", name="HuggingFaceH4")\],
    domains=\["repo", "discussion"\],
    secret="my-secret",
    disabled=False,
)

#### dataset\_info

[](#huggingface_hub.HfApi.dataset_info)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2523)

( repo\_id: strrevision: Optional\[str\] = Nonetimeout: Optional\[float\] = Nonefiles\_metadata: bool = Falseexpand: Optional\[List\[ExpandDatasetProperty\_T\]\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[hf\_api.DatasetInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.DatasetInfo)

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.dataset_info.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.dataset_info.revision)**revision** (`str`, _optional_) — The revision of the dataset repository from which to get the information.
*   [](#huggingface_hub.HfApi.dataset_info.timeout)**timeout** (`float`, _optional_) — Whether to set a timeout for the request to the Hub.
*   [](#huggingface_hub.HfApi.dataset_info.files_metadata)**files\_metadata** (`bool`, _optional_) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to `False`.
*   [](#huggingface_hub.HfApi.dataset_info.expand)**expand** (`List[ExpandDatasetProperty_T]`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `files_metadata` is passed. Possible values are `"author"`, `"cardData"`, `"citation"`, `"createdAt"`, `"disabled"`, `"description"`, `"downloads"`, `"downloadsAllTime"`, `"gated"`, `"lastModified"`, `"likes"`, `"paperswithcode_id"`, `"private"`, `"siblings"`, `"sha"`, `"tags"`, `"trendingScore"`,`"usedStorage"` and `"resourceGroup"`.
*   [](#huggingface_hub.HfApi.dataset_info.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[hf\_api.DatasetInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.DatasetInfo)

export const metadata = 'undefined';

The dataset repository information.

Get info on one specific dataset on huggingface.co.

Dataset can be private if you pass an acceptable token.

Raises the following errors:

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) If the revision to download from cannot be found.

#### delete\_branch

[](#huggingface_hub.HfApi.delete_branch)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5719)

( repo\_id: strbranch: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_branch.repo_id)**repo\_id** (`str`) — The repository in which a branch will be deleted. Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.delete_branch.branch)**branch** (`str`) — The name of the branch to delete.
*   [](#huggingface_hub.HfApi.delete_branch.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.delete_branch.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if creating a branch on a dataset or space, `None` or `"model"` if tagging a model. Default is `None`.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — If trying to delete a protected branch. Ex: `main` cannot be deleted.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — If trying to delete a branch that does not exist.

Delete a branch from a repo on the Hub.

#### delete\_collection

[](#huggingface_hub.HfApi.delete_collection)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8035)

( collection\_slug: strmissing\_ok: bool = Falsetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_collection.collection_slug)**collection\_slug** (`str`) — Slug of the collection to delete. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
*   [](#huggingface_hub.HfApi.delete_collection.missing_ok)**missing\_ok** (`bool`, _optional_) — If `True`, do not raise an error if collection doesn’t exists.
*   [](#huggingface_hub.HfApi.delete_collection.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Delete a collection on the Hub.

[](#huggingface_hub.HfApi.delete_collection.example)

Example:

Copied

\>>> from huggingface\_hub import delete\_collection
\>>> collection = delete\_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing\_ok=True)

This is a non-revertible action. A deleted collection cannot be restored.

#### delete\_collection\_item

[](#huggingface_hub.HfApi.delete_collection_item)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8211)

( collection\_slug: stritem\_object\_id: strmissing\_ok: bool = Falsetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_collection_item.collection_slug)**collection\_slug** (`str`) — Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
*   [](#huggingface_hub.HfApi.delete_collection_item.item_object_id)**item\_object\_id** (`str`) — ID of the item in the collection. This is not the id of the item on the Hub (repo\_id or paper id). It must be retrieved from a [CollectionItem](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.CollectionItem) object. Example: `collection.items[0].item_object_id`.
*   [](#huggingface_hub.HfApi.delete_collection_item.missing_ok)**missing\_ok** (`bool`, _optional_) — If `True`, do not raise an error if item doesn’t exists.
*   [](#huggingface_hub.HfApi.delete_collection_item.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Delete an item from a collection.

[](#huggingface_hub.HfApi.delete_collection_item.example)

Example:

Copied

\>>> from huggingface\_hub import get\_collection, delete\_collection\_item

\# Get collection first
\>>> collection = get\_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")

\# Delete item based on its ID
\>>> delete\_collection\_item(
...     collection\_slug="TheBloke/recent-models-64f9a55bb3115b4f513ec026",
...     item\_object\_id=collection.items\[-1\].item\_object\_id,
... )

#### delete\_file

[](#huggingface_hub.HfApi.delete_file)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4737)

( path\_in\_repo: strrepo\_id: strtoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecommit\_message: Optional\[str\] = Nonecommit\_description: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Noneparent\_commit: Optional\[str\] = None )

Expand 9 parameters

Parameters

*   [](#huggingface_hub.HfApi.delete_file.path_in_repo)**path\_in\_repo** (`str`) — Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
*   [](#huggingface_hub.HfApi.delete_file.repo_id)**repo\_id** (`str`) — The repository from which the file will be deleted, for example: `"username/custom_transformers"`
*   [](#huggingface_hub.HfApi.delete_file.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.delete_file.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if the file is in a dataset or space, `None` or `"model"` if in a model. Default is `None`.
*   [](#huggingface_hub.HfApi.delete_file.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.delete_file.commit_message)**commit\_message** (`str`, _optional_) — The summary / title / first line of the generated commit. Defaults to `f"Delete {path_in_repo} with huggingface_hub"`.
*   [](#huggingface_hub.HfApi.delete_file.commit_description)**commit\_description** (`str` _optional_) — The description of the generated commit
*   [](#huggingface_hub.HfApi.delete_file.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened against this branch. If `revision` is set and is not a branch name (example: a commit oid), an `RevisionNotFoundError` is returned by the server.
*   [](#huggingface_hub.HfApi.delete_file.parent_commit)**parent\_commit** (`str`, _optional_) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.

Deletes a file in the given repo.

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) If the revision to download from cannot be found.
*   [EntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError) If the file to download cannot be found.

#### delete\_files

[](#huggingface_hub.HfApi.delete_files)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4827)

( repo\_id: strdelete\_patterns: List\[str\]token: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecommit\_message: Optional\[str\] = Nonecommit\_description: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Noneparent\_commit: Optional\[str\] = None )

Expand 9 parameters

Parameters

*   [](#huggingface_hub.HfApi.delete_files.repo_id)**repo\_id** (`str`) — The repository from which the folder will be deleted, for example: `"username/custom_transformers"`
*   [](#huggingface_hub.HfApi.delete_files.delete_patterns)**delete\_patterns** (`List[str]`) — List of files or folders to delete. Each string can either be a file path, a folder path or a Unix shell-style wildcard. E.g. `["file.txt", "folder/", "data/*.parquet"]`
*   [](#huggingface_hub.HfApi.delete_files.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`. to the stored token.
*   [](#huggingface_hub.HfApi.delete_files.repo_type)**repo\_type** (`str`, _optional_) — Type of the repo to delete files from. Can be `"model"`, `"dataset"` or `"space"`. Defaults to `"model"`.
*   [](#huggingface_hub.HfApi.delete_files.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.delete_files.commit_message)**commit\_message** (`str`, _optional_) — The summary (first line) of the generated commit. Defaults to `f"Delete files using huggingface_hub"`.
*   [](#huggingface_hub.HfApi.delete_files.commit_description)**commit\_description** (`str` _optional_) — The description of the generated commit.
*   [](#huggingface_hub.HfApi.delete_files.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened against this branch. If `revision` is set and is not a branch name (example: a commit oid), an `RevisionNotFoundError` is returned by the server.
*   [](#huggingface_hub.HfApi.delete_files.parent_commit)**parent\_commit** (`str`, _optional_) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.

Delete files from a repository on the Hub.

If a folder path is provided, the entire folder is deleted as well as all files it contained.

#### delete\_folder

[](#huggingface_hub.HfApi.delete_folder)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4903)

( path\_in\_repo: strrepo\_id: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecommit\_message: Optional\[str\] = Nonecommit\_description: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Noneparent\_commit: Optional\[str\] = None )

Expand 9 parameters

Parameters

*   [](#huggingface_hub.HfApi.delete_folder.path_in_repo)**path\_in\_repo** (`str`) — Relative folder path in the repo, for example: `"checkpoints/1fec34a"`.
*   [](#huggingface_hub.HfApi.delete_folder.repo_id)**repo\_id** (`str`) — The repository from which the folder will be deleted, for example: `"username/custom_transformers"`
*   [](#huggingface_hub.HfApi.delete_folder.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`. to the stored token.
*   [](#huggingface_hub.HfApi.delete_folder.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if the folder is in a dataset or space, `None` or `"model"` if in a model. Default is `None`.
*   [](#huggingface_hub.HfApi.delete_folder.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.delete_folder.commit_message)**commit\_message** (`str`, _optional_) — The summary / title / first line of the generated commit. Defaults to `f"Delete folder {path_in_repo} with huggingface_hub"`.
*   [](#huggingface_hub.HfApi.delete_folder.commit_description)**commit\_description** (`str` _optional_) — The description of the generated commit.
*   [](#huggingface_hub.HfApi.delete_folder.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened against this branch. If `revision` is set and is not a branch name (example: a commit oid), an `RevisionNotFoundError` is returned by the server.
*   [](#huggingface_hub.HfApi.delete_folder.parent_commit)**parent\_commit** (`str`, _optional_) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.

Deletes a folder in the given repo.

Simple wrapper around [create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) method.

#### delete\_inference\_endpoint

[](#huggingface_hub.HfApi.delete_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7643)

( name: strnamespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_inference_endpoint.name)**name** (`str`) — The name of the Inference Endpoint to delete.
*   [](#huggingface_hub.HfApi.delete_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace in which the Inference Endpoint is located. Defaults to the current user.
*   [](#huggingface_hub.HfApi.delete_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Delete an Inference Endpoint.

This operation is not reversible. If you don’t want to be charged for an Inference Endpoint, it is preferable to pause it with [pause\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint) or scale it to zero with [scale\_to\_zero\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint).

For convenience, you can also delete an Inference Endpoint using [InferenceEndpoint.delete()](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.delete).

#### delete\_repo

[](#huggingface_hub.HfApi.delete_repo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3531)

( repo\_id: strtoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = Nonemissing\_ok: bool = False )

Parameters

*   [](#huggingface_hub.HfApi.delete_repo.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.delete_repo.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.delete_repo.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model.
*   [](#huggingface_hub.HfApi.delete_repo.missing_ok)**missing\_ok** (`bool`, _optional_, defaults to `False`) — If `True`, do not raise an error if repo does not exist.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If the repository to delete from cannot be found and `missing_ok` is set to False (default).

Delete a repo from the HuggingFace Hub. CAUTION: this is irreversible.

#### delete\_space\_secret

[](#huggingface_hub.HfApi.delete_space_secret)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6745)

( repo\_id: strkey: strtoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_space_secret.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.delete_space_secret.key)**key** (`str`) — Secret key. Example: `"GITHUB_API_KEY"`.
*   [](#huggingface_hub.HfApi.delete_space_secret.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Deletes a secret from a Space.

Secrets allow to set secret keys or tokens to a Space without hardcoding them. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

#### delete\_space\_storage

[](#huggingface_hub.HfApi.delete_space_storage)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7229)

( repo\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Parameters

*   [](#huggingface_hub.HfApi.delete_space_storage.repo_id)**repo\_id** (`str`) — ID of the Space to update. Example: `"open-llm-leaderboard/open_llm_leaderboard"`.
*   [](#huggingface_hub.HfApi.delete_space_storage.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about a Space including Space stage and hardware.

Raises

export const metadata = 'undefined';

`BadRequestError`

export const metadata = 'undefined';

*   `BadRequestError` — If space has no persistent storage.

Delete persistent storage for a Space.

#### delete\_space\_variable

[](#huggingface_hub.HfApi.delete_space_variable)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6834)

( repo\_id: strkey: strtoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_space_variable.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.delete_space_variable.key)**key** (`str`) — Variable key. Example: `"MODEL_REPO_ID"`
*   [](#huggingface_hub.HfApi.delete_space_variable.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Deletes a variable from a Space.

Variables allow to set environment variables to a Space without hardcoding them. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables)

#### delete\_tag

[](#huggingface_hub.HfApi.delete_tag)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5845)

( repo\_id: strtag: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.HfApi.delete_tag.repo_id)**repo\_id** (`str`) — The repository in which a tag will be deleted. Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.delete_tag.tag)**tag** (`str`) — The name of the tag to delete.
*   [](#huggingface_hub.HfApi.delete_tag.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.delete_tag.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if tagging a dataset or space, `None` or `"model"` if tagging a model. Default is `None`.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If tag is not found.

Delete a tag from a repo on the Hub.

#### delete\_webhook

[](#huggingface_hub.HfApi.delete_webhook)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9057)

( webhook\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`None`

Parameters

*   [](#huggingface_hub.HfApi.delete_webhook.webhook_id)**webhook\_id** (`str`) — The unique identifier of the webhook to delete.
*   [](#huggingface_hub.HfApi.delete_webhook.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`None`

Delete a webhook.

[](#huggingface_hub.HfApi.delete_webhook.example)

Example:

Copied

\>>> from huggingface\_hub import delete\_webhook
\>>> delete\_webhook("654bbbc16f2ec14d77f109cc")

#### disable\_webhook

[](#huggingface_hub.HfApi.disable_webhook)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9006)

( webhook\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

Parameters

*   [](#huggingface_hub.HfApi.disable_webhook.webhook_id)**webhook\_id** (`str`) — The unique identifier of the webhook to disable.
*   [](#huggingface_hub.HfApi.disable_webhook.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

export const metadata = 'undefined';

Info about the disabled webhook.

Disable a webhook (makes it “disabled”).

[](#huggingface_hub.HfApi.disable_webhook.example)

Example:

Copied

\>>> from huggingface\_hub import disable\_webhook
\>>> disabled\_webhook = disable\_webhook("654bbbc16f2ec14d77f109cc")
\>>> disabled\_webhook
WebhookInfo(
    id\="654bbbc16f2ec14d77f109cc",
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    watched=\[WebhookWatchedItem(type\="user", name="julien-c"), WebhookWatchedItem(type\="org", name="HuggingFaceH4")\],
    domains=\["repo", "discussion"\],
    secret="my-secret",
    disabled=True,
)

#### duplicate\_space

[](#huggingface_hub.HfApi.duplicate_space)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7072)

( from\_id: strto\_id: Optional\[str\] = Noneprivate: Optional\[bool\] = Nonetoken: Union\[bool, str, None\] = Noneexist\_ok: bool = Falsehardware: Optional\[SpaceHardware\] = Nonestorage: Optional\[SpaceStorage\] = Nonesleep\_time: Optional\[int\] = Nonesecrets: Optional\[List\[Dict\[str, str\]\]\] = Nonevariables: Optional\[List\[Dict\[str, str\]\]\] = None ) → export const metadata = 'undefined';[RepoUrl](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.RepoUrl)

Expand 10 parameters

Parameters

*   [](#huggingface_hub.HfApi.duplicate_space.from_id)**from\_id** (`str`) — ID of the Space to duplicate. Example: `"pharma/CLIP-Interrogator"`.
*   [](#huggingface_hub.HfApi.duplicate_space.to_id)**to\_id** (`str`, _optional_) — ID of the new Space. Example: `"dog/CLIP-Interrogator"`. If not provided, the new Space will have the same name as the original Space, but in your account.
*   [](#huggingface_hub.HfApi.duplicate_space.private)**private** (`bool`, _optional_) — Whether the new Space should be private or not. Defaults to the same privacy as the original Space.
*   [](#huggingface_hub.HfApi.duplicate_space.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.duplicate_space.exist_ok)**exist\_ok** (`bool`, _optional_, defaults to `False`) — If `True`, do not raise an error if repo already exists.
*   [](#huggingface_hub.HfApi.duplicate_space.hardware)**hardware** (`SpaceHardware` or `str`, _optional_) — Choice of Hardware. Example: `"t4-medium"`. See [SpaceHardware](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceHardware) for a complete list.
*   [](#huggingface_hub.HfApi.duplicate_space.storage)**storage** (`SpaceStorage` or `str`, _optional_) — Choice of persistent storage tier. Example: `"small"`. See [SpaceStorage](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceStorage) for a complete list.
*   [](#huggingface_hub.HfApi.duplicate_space.sleep_time)**sleep\_time** (`int`, _optional_) — Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don’t want your Space to sleep (default behavior for upgraded hardware). For free hardware, you can’t configure the sleep time (value is fixed to 48 hours of inactivity). See [https://huggingface.co/docs/hub/spaces-gpus#sleep-time](https://huggingface.co/docs/hub/spaces-gpus#sleep-time) for more details.
*   [](#huggingface_hub.HfApi.duplicate_space.secrets)**secrets** (`List[Dict[str, str]]`, _optional_) — A list of secret keys to set in your Space. Each item is in the form `{"key": ..., "value": ..., "description": ...}` where description is optional. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).
*   [](#huggingface_hub.HfApi.duplicate_space.variables)**variables** (`List[Dict[str, str]]`, _optional_) — A list of public environment variables to set in your Space. Each item is in the form `{"key": ..., "value": ..., "description": ...}` where description is optional. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables).

Returns

export const metadata = 'undefined';

[RepoUrl](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.RepoUrl)

export const metadata = 'undefined';

URL to the newly created repo. Value is a subclass of `str` containing attributes like `endpoint`, `repo_type` and `repo_id`.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or `HTTPError`

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If one of `from_id` or `to_id` cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — If the HuggingFace API returned an error

Duplicate a Space.

Programmatically duplicate a Space. The new Space will be created in your account and will be in the same state as the original Space (running or paused). You can duplicate a Space no matter the current state of a Space.

[](#huggingface_hub.HfApi.duplicate_space.example)

Example:

Copied

\>>> from huggingface\_hub import duplicate\_space

\# Duplicate a Space to your account
\>>> duplicate\_space("multimodalart/dreambooth-training")
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)

\# Can set custom destination id and visibility flag.
\>>> duplicate\_space("multimodalart/dreambooth-training", to\_id="my-dreambooth", private=True)
RepoUrl('https://huggingface.co/spaces/nateraw/my-dreambooth',...)

#### edit\_discussion\_comment

[](#huggingface_hub.HfApi.edit_discussion_comment)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6581)

( repo\_id: strdiscussion\_num: intcomment\_id: strnew\_content: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None ) → export const metadata = 'undefined';[DiscussionComment](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionComment)

Parameters

*   [](#huggingface_hub.HfApi.edit_discussion_comment.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.edit_discussion_comment.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.edit_discussion_comment.comment_id)**comment\_id** (`str`) — The ID of the comment to edit.
*   [](#huggingface_hub.HfApi.edit_discussion_comment.new_content)**new\_content** (`str`) — The new content of the comment. Comments support markdown formatting.
*   [](#huggingface_hub.HfApi.edit_discussion_comment.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.edit_discussion_comment.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[DiscussionComment](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionComment)

export const metadata = 'undefined';

the edited comment

Edits a comment on a Discussion / Pull Request.

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### enable\_webhook

[](#huggingface_hub.HfApi.enable_webhook)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8955)

( webhook\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

Parameters

*   [](#huggingface_hub.HfApi.enable_webhook.webhook_id)**webhook\_id** (`str`) — The unique identifier of the webhook to enable.
*   [](#huggingface_hub.HfApi.enable_webhook.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

export const metadata = 'undefined';

Info about the enabled webhook.

Enable a webhook (makes it “active”).

[](#huggingface_hub.HfApi.enable_webhook.example)

Example:

Copied

\>>> from huggingface\_hub import enable\_webhook
\>>> enabled\_webhook = enable\_webhook("654bbbc16f2ec14d77f109cc")
\>>> enabled\_webhook
WebhookInfo(
    id\="654bbbc16f2ec14d77f109cc",
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    watched=\[WebhookWatchedItem(type\="user", name="julien-c"), WebhookWatchedItem(type\="org", name="HuggingFaceH4")\],
    domains=\["repo", "discussion"\],
    secret="my-secret",
    disabled=False,
)

#### file\_exists

[](#huggingface_hub.HfApi.file_exists)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2834)

( repo\_id: strfilename: strrepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.file_exists.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.file_exists.filename)**filename** (`str`) — The name of the file to check, for example: `"config.json"`
*   [](#huggingface_hub.HfApi.file_exists.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space, `None` or `"model"` if getting repository info from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.file_exists.revision)**revision** (`str`, _optional_) — The revision of the repository from which to get the information. Defaults to `"main"` branch.
*   [](#huggingface_hub.HfApi.file_exists.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Checks if a file exists in a repository on the Hugging Face Hub.

[](#huggingface_hub.HfApi.file_exists.example)

Examples:

Copied

\>>> from huggingface\_hub import file\_exists
\>>> file\_exists("bigcode/starcoder", "config.json")
True
\>>> file\_exists("bigcode/starcoder", "not-a-file")
False
\>>> file\_exists("bigcode/not-a-repo", "config.json")
False

#### get\_collection

[](#huggingface_hub.HfApi.get_collection)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7860)

( collection\_slug: strtoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.get_collection.collection_slug)**collection\_slug** (`str`) — Slug of the collection of the Hub. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
*   [](#huggingface_hub.HfApi.get_collection.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Gets information about a Collection on the Hub.

Returns: [Collection](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.Collection)

[](#huggingface_hub.HfApi.get_collection.example)

Example:

Copied

\>>> from huggingface\_hub import get\_collection
\>>> collection = get\_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")
\>>> collection.title
'Recent models'
\>>> len(collection.items)
37
\>>> collection.items\[0\]
CollectionItem(
    item\_object\_id='651446103cd773a050bf64c2',
    item\_id='TheBloke/U-Amethyst-20B-AWQ',
    item\_type='model',
    position=88,
    note=None
)

#### get\_dataset\_tags

[](#huggingface_hub.HfApi.get_dataset_tags)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1729)

( )

List all valid dataset tags as a nested namespace object.

#### get\_discussion\_details

[](#huggingface_hub.HfApi.get_discussion_details)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6040)

( repo\_id: strdiscussion\_num: intrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.get_discussion_details.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.get_discussion_details.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.get_discussion_details.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.get_discussion_details.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Fetches a Discussion’s / Pull Request ‘s details from the Hub.

Returns: [DiscussionWithDetails](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionWithDetails)

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### get\_full\_repo\_name

[](#huggingface_hub.HfApi.get_full_repo_name)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5894)

( model\_id: strorganization: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`str`

Parameters

*   [](#huggingface_hub.HfApi.get_full_repo_name.model_id)**model\_id** (`str`) — The name of the model.
*   [](#huggingface_hub.HfApi.get_full_repo_name.organization)**organization** (`str`, _optional_) — If passed, the repository name will be in the organization namespace instead of the user namespace.
*   [](#huggingface_hub.HfApi.get_full_repo_name.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`str`

export const metadata = 'undefined';

The repository name in the user’s namespace ({username}/{model\_id}) if no organization is passed, and under the organization namespace ({organization}/{model\_id}) otherwise.

Returns the repository name for a given model ID and optional organization.

#### get\_hf\_file\_metadata

[](#huggingface_hub.HfApi.get_hf_file_metadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5090)

( url: strtoken: Union\[bool, str, None\] = Noneproxies: Optional\[Dict\] = Nonetimeout: Optional\[float\] = 10 )

Parameters

*   [](#huggingface_hub.HfApi.get_hf_file_metadata.url)**url** (`str`) — File url, for example returned by [hf\_hub\_url()](/docs/huggingface_hub/v0.29.2/en/package_reference/file_download#huggingface_hub.hf_hub_url).
*   [](#huggingface_hub.HfApi.get_hf_file_metadata.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.get_hf_file_metadata.proxies)**proxies** (`dict`, _optional_) — Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
*   [](#huggingface_hub.HfApi.get_hf_file_metadata.timeout)**timeout** (`float`, _optional_, defaults to 10) — How many seconds to wait for the server to send metadata before giving up.

Fetch metadata of a file versioned on the Hub for a given url.

#### get\_inference\_endpoint

[](#huggingface_hub.HfApi.get_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7488)

( name: strnamespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Parameters

*   [](#huggingface_hub.HfApi.get_inference_endpoint.name)**name** (`str`) — The name of the Inference Endpoint to retrieve information about.
*   [](#huggingface_hub.HfApi.get_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace in which the Inference Endpoint is located. Defaults to the current user.
*   [](#huggingface_hub.HfApi.get_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

information about the requested Inference Endpoint.

Get information about an Inference Endpoint.

[](#huggingface_hub.HfApi.get_inference_endpoint.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> endpoint = api.get\_inference\_endpoint("my-text-to-image")
\>>> endpoint
InferenceEndpoint(name='my-text-to-image', ...)

\# Get status
\>>> endpoint.status
'running'
\>>> endpoint.url
'https://my-text-to-image.region.vendor.endpoints.huggingface.cloud'

\# Run inference
\>>> endpoint.client.text\_to\_image(...)

#### get\_model\_tags

[](#huggingface_hub.HfApi.get_model_tags)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1720)

( )

List all valid model tags as a nested namespace object

#### get\_paths\_info

[](#huggingface_hub.HfApi.get_paths_info)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3219)

( repo\_id: strpaths: Union\[List\[str\], str\]expand: bool = Falserevision: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None ) → export const metadata = 'undefined';`List[Union[RepoFile, RepoFolder]]`

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.get_paths_info.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.get_paths_info.paths)**paths** (`Union[List[str], str]`, _optional_) — The paths to get information about. If a path do not exist, it is ignored without raising an exception.
*   [](#huggingface_hub.HfApi.get_paths_info.expand)**expand** (`bool`, _optional_, defaults to `False`) — Whether to fetch more information about the paths (e.g. last commit and files’ security scan results). This operation is more expensive for the server so only 50 results are returned per page (instead of 1000). As pagination is implemented in `huggingface_hub`, this is transparent for you except for the time it takes to get the results.
*   [](#huggingface_hub.HfApi.get_paths_info.revision)**revision** (`str`, _optional_) — The revision of the repository from which to get the information. Defaults to `"main"` branch.
*   [](#huggingface_hub.HfApi.get_paths_info.repo_type)**repo\_type** (`str`, _optional_) — The type of the repository from which to get the information (`"model"`, `"dataset"` or `"space"`. Defaults to `"model"`.
*   [](#huggingface_hub.HfApi.get_paths_info.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`List[Union[RepoFile, RepoFolder]]`

export const metadata = 'undefined';

The information about the paths, as a list of `RepoFile` and `RepoFolder` objects.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If revision is not found (error 404) on the repo.

Get information about a repo’s paths.

[](#huggingface_hub.HfApi.get_paths_info.example)

Example:

Copied

\>>> from huggingface\_hub import get\_paths\_info
\>>> paths\_info = get\_paths\_info("allenai/c4", \["README.md", "en"\], repo\_type="dataset")
\>>> paths\_info
\[
    RepoFile(path='README.md', size=2379, blob\_id='f84cb4c97182890fc1dbdeaf1a6a468fd27b4fff', lfs=None, last\_commit=None, security=None),
    RepoFolder(path='en', tree\_id='dc943c4c40f53d02b31ced1defa7e5f438d5862e', last\_commit=None)
\]

#### get\_repo\_discussions

[](#huggingface_hub.HfApi.get_repo_discussions)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5932)

( repo\_id: strauthor: Optional\[str\] = Nonediscussion\_type: Optional\[constants.DiscussionTypeFilter\] = Nonediscussion\_status: Optional\[constants.DiscussionStatusFilter\] = Nonerepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterator[Discussion]`

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.get_repo_discussions.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.get_repo_discussions.author)**author** (`str`, _optional_) — Pass a value to filter by discussion author. `None` means no filter. Default is `None`.
*   [](#huggingface_hub.HfApi.get_repo_discussions.discussion_type)**discussion\_type** (`str`, _optional_) — Set to `"pull_request"` to fetch only pull requests, `"discussion"` to fetch only discussions. Set to `"all"` or `None` to fetch both. Default is `None`.
*   [](#huggingface_hub.HfApi.get_repo_discussions.discussion_status)**discussion\_status** (`str`, _optional_) — Set to `"open"` (respectively `"closed"`) to fetch only open (respectively closed) discussions. Set to `"all"` or `None` to fetch both. Default is `None`.
*   [](#huggingface_hub.HfApi.get_repo_discussions.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if fetching from a dataset or space, `None` or `"model"` if fetching from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.get_repo_discussions.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterator[Discussion]`

export const metadata = 'undefined';

An iterator of [Discussion](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.Discussion) objects.

Fetches Discussions and Pull Requests for the given repo.

Example:

[](#huggingface_hub.HfApi.get_repo_discussions.example)

Collecting all discussions of a repo in a list:

Copied

\>>> from huggingface\_hub import get\_repo\_discussions
\>>> discussions\_list = list(get\_repo\_discussions(repo\_id="bert-base-uncased"))

[](#huggingface_hub.HfApi.get_repo_discussions.example-2)

Iterating over discussions of a repo:

Copied

\>>> from huggingface\_hub import get\_repo\_discussions
\>>> for discussion in get\_repo\_discussions(repo\_id="bert-base-uncased"):
...     print(discussion.num, discussion.title)

#### get\_safetensors\_metadata

[](#huggingface_hub.HfApi.get_safetensors_metadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5396)

( repo\_id: strrepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`SafetensorsRepoMetadata`

Expand 4 parameters

Parameters

*   [](#huggingface_hub.HfApi.get_safetensors_metadata.repo_id)**repo\_id** (`str`) — A user or an organization name and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.get_safetensors_metadata.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if the file is in a dataset or space, `None` or `"model"` if in a model. Default is `None`.
*   [](#huggingface_hub.HfApi.get_safetensors_metadata.revision)**revision** (`str`, _optional_) — The git revision to fetch the file from. Can be a branch name, a tag, or a commit hash. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.get_safetensors_metadata.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`SafetensorsRepoMetadata`

export const metadata = 'undefined';

information related to safetensors repo.

Raises

export const metadata = 'undefined';

`NotASafetensorsRepoError` or `SafetensorsParsingError`

export const metadata = 'undefined';

*   `NotASafetensorsRepoError` — If the repo is not a safetensors repo i.e. doesn’t have either a `model.safetensors` or a `model.safetensors.index.json` file.
*   `SafetensorsParsingError` — If a safetensors file header couldn’t be parsed correctly.

Parse metadata for a safetensors repo on the Hub.

We first check if the repo has a single safetensors file or a sharded safetensors repo. If it’s a single safetensors file, we parse the metadata from this file. If it’s a sharded safetensors repo, we parse the metadata from the index file and then parse the metadata from each shard.

To parse metadata from a single safetensors file, use [parse\_safetensors\_file\_metadata()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.parse_safetensors_file_metadata).

For more details regarding the safetensors format, check out [https://huggingface.co/docs/safetensors/index#format](https://huggingface.co/docs/safetensors/index#format).

[](#huggingface_hub.HfApi.get_safetensors_metadata.example)

Example:

Copied

\# Parse repo with single weights file
\>>> metadata = get\_safetensors\_metadata("bigscience/bloomz-560m")
\>>> metadata
SafetensorsRepoMetadata(
    metadata=None,
    sharded=False,
    weight\_map={'h.0.input\_layernorm.bias': 'model.safetensors', ...},
    files\_metadata={'model.safetensors': SafetensorsFileMetadata(...)}
)
\>>> metadata.files\_metadata\["model.safetensors"\].metadata
{'format': 'pt'}

\# Parse repo with sharded model
\>>> metadata = get\_safetensors\_metadata("bigscience/bloom")
Parse safetensors files: 100%|██████████████████████████████████████████| 72/72 \[00:12<00:00,  5.78it/s\]
\>>> metadata
SafetensorsRepoMetadata(metadata={'total\_size': 352494542848}, sharded=True, weight\_map={...}, files\_metadata={...})
\>>> len(metadata.files\_metadata)
72  \# All safetensors files have been fetched

\# Parse repo with sharded model
\>>> get\_safetensors\_metadata("runwayml/stable-diffusion-v1-5")
NotASafetensorsRepoError: 'runwayml/stable-diffusion-v1-5' is not a safetensors repo. Couldn't find 'model.safetensors.index.json' or 'model.safetensors' files.

#### get\_space\_runtime

[](#huggingface_hub.HfApi.get_space_runtime)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6862)

( repo\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Parameters

*   [](#huggingface_hub.HfApi.get_space_runtime.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.get_space_runtime.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about a Space including Space stage and hardware.

Gets runtime information about a Space.

#### get\_space\_variables

[](#huggingface_hub.HfApi.get_space_variables)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6770)

( repo\_id: strtoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.get_space_variables.repo_id)**repo\_id** (`str`) — ID of the repo to query. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.get_space_variables.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Gets all variables from a Space.

Variables allow to set environment variables to a Space without hardcoding them. For more details, see [https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables)

#### get\_token\_permission

[](#huggingface_hub.HfApi.get_token_permission)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1680)

( token: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Literal["read", "write", "fineGrained", None]`

Parameters

*   [](#huggingface_hub.HfApi.get_token_permission.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Literal["read", "write", "fineGrained", None]`

export const metadata = 'undefined';

Permission granted by the token (“read” or “write”). Returns `None` if no token passed, if token is invalid or if role is not returned by the server. This typically happens when the token is an OAuth token.

Check if a given `token` is valid and return its permissions.

This method is deprecated and will be removed in version 1.0. Permissions are more complex than when `get_token_permission` was first introduced. OAuth and fine-grain tokens allows for more detailed permissions. If you need to know the permissions associated with a token, please use `whoami` and check the `'auth'` key.

For more details about tokens, please refer to [https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

#### get\_user\_overview

[](#huggingface_hub.HfApi.get_user_overview)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9256)

( username: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`User`

Parameters

*   [](#huggingface_hub.HfApi.get_user_overview.username)**username** (`str`) — Username of the user to get an overview of.
*   [](#huggingface_hub.HfApi.get_user_overview.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`User`

export const metadata = 'undefined';

A [User](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.User) object with the user’s overview.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 If the user does not exist on the Hub.

Get an overview of a user on the Hub.

#### get\_webhook

[](#huggingface_hub.HfApi.get_webhook)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8702)

( webhook\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

Parameters

*   [](#huggingface_hub.HfApi.get_webhook.webhook_id)**webhook\_id** (`str`) — The unique identifier of the webhook to get.
*   [](#huggingface_hub.HfApi.get_webhook.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

export const metadata = 'undefined';

Info about the webhook.

Get a webhook by its id.

[](#huggingface_hub.HfApi.get_webhook.example)

Example:

Copied

\>>> from huggingface\_hub import get\_webhook
\>>> webhook = get\_webhook("654bbbc16f2ec14d77f109cc")
\>>> print(webhook)
WebhookInfo(
    id\="654bbbc16f2ec14d77f109cc",
    watched=\[WebhookWatchedItem(type\="user", name="julien-c"), WebhookWatchedItem(type\="org", name="HuggingFaceH4")\],
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    secret="my-secret",
    domains=\["repo", "discussion"\],
    disabled=False,
)

#### grant\_access

[](#huggingface_hub.HfApi.grant_access)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8647)

( repo\_id: struser: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Expand 4 parameters

Parameters

*   [](#huggingface_hub.HfApi.grant_access.repo_id)**repo\_id** (`str`) — The id of the repo to grant access to.
*   [](#huggingface_hub.HfApi.grant_access.user)**user** (`str`) — The username of the user to grant access.
*   [](#huggingface_hub.HfApi.grant_access.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to grant access to. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.grant_access.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the user already has access to the repo.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user does not exist on the Hub.

Grant access to a user for a given gated repo.

Granting access don’t require for the user to send an access request by themselves. The user is automatically added to the accepted list meaning they can download the files You can revoke the granted access at any time using [cancel\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request) or [reject\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request).

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

#### hf\_hub\_download

[](#huggingface_hub.HfApi.hf_hub_download)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5131)

( repo\_id: strfilename: strsubfolder: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecache\_dir: Union\[str, Path, None\] = Nonelocal\_dir: Union\[str, Path, None\] = Noneforce\_download: bool = Falseproxies: Optional\[Dict\] = Noneetag\_timeout: float = 10token: Union\[bool, str, None\] = Nonelocal\_files\_only: bool = Falseresume\_download: Optional\[bool\] = Noneforce\_filename: Optional\[str\] = Nonelocal\_dir\_use\_symlinks: Union\[bool, Literal\['auto'\]\] = 'auto' ) → export const metadata = 'undefined';`str`

Expand 12 parameters

Parameters

*   [](#huggingface_hub.HfApi.hf_hub_download.repo_id)**repo\_id** (`str`) — A user or an organization name and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.hf_hub_download.filename)**filename** (`str`) — The name of the file in the repo.
*   [](#huggingface_hub.HfApi.hf_hub_download.subfolder)**subfolder** (`str`, _optional_) — An optional value corresponding to a folder inside the repository.
*   [](#huggingface_hub.HfApi.hf_hub_download.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if downloading from a dataset or space, `None` or `"model"` if downloading from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.hf_hub_download.revision)**revision** (`str`, _optional_) — An optional Git revision id which can be a branch name, a tag, or a commit hash.
*   [](#huggingface_hub.HfApi.hf_hub_download.cache_dir)**cache\_dir** (`str`, `Path`, _optional_) — Path to the folder where cached files are stored.
*   [](#huggingface_hub.HfApi.hf_hub_download.local_dir)**local\_dir** (`str` or `Path`, _optional_) — If provided, the downloaded file will be placed under this directory.
*   [](#huggingface_hub.HfApi.hf_hub_download.force_download)**force\_download** (`bool`, _optional_, defaults to `False`) — Whether the file should be downloaded even if it already exists in the local cache.
*   [](#huggingface_hub.HfApi.hf_hub_download.proxies)**proxies** (`dict`, _optional_) — Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
*   [](#huggingface_hub.HfApi.hf_hub_download.etag_timeout)**etag\_timeout** (`float`, _optional_, defaults to `10`) — When fetching ETag, how many seconds to wait for the server to send data before giving up which is passed to `requests.request`.
*   [](#huggingface_hub.HfApi.hf_hub_download.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.hf_hub_download.local_files_only)**local\_files\_only** (`bool`, _optional_, defaults to `False`) — If `True`, avoid downloading the file and return the path to the local cached file if it exists.

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

[](#huggingface_hub.HfApi.hf_hub_download.example)

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

#### hide\_discussion\_comment

[](#huggingface_hub.HfApi.hide_discussion_comment)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6641)

( repo\_id: strdiscussion\_num: intcomment\_id: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None ) → export const metadata = 'undefined';[DiscussionComment](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionComment)

Parameters

*   [](#huggingface_hub.HfApi.hide_discussion_comment.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.hide_discussion_comment.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.hide_discussion_comment.comment_id)**comment\_id** (`str`) — The ID of the comment to edit.
*   [](#huggingface_hub.HfApi.hide_discussion_comment.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.hide_discussion_comment.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[DiscussionComment](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionComment)

export const metadata = 'undefined';

the hidden comment

Hides a comment on a Discussion / Pull Request.

Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### list\_accepted\_access\_requests

[](#huggingface_hub.HfApi.list_accepted_access_requests)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8331)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`List[AccessRequest]`

Expand 3 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_accepted_access_requests.repo_id)**repo\_id** (`str`) — The id of the repo to get access requests for.
*   [](#huggingface_hub.HfApi.list_accepted_access_requests.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to get access requests for. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.list_accepted_access_requests.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`List[AccessRequest]`

export const metadata = 'undefined';

A list of `AccessRequest` objects. Each time contains a `username`, `email`, `status` and `timestamp` attribute. If the gated repo has a custom form, the `fields` attribute will be populated with user’s answers.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.

Get accepted access requests for a given gated repo.

An accepted request means the user has requested access to the repo and the request has been accepted. The user can download any file of the repo. If the approval mode is automatic, this list should contains by default all requests. Accepted requests can be cancelled or rejected at any time using [cancel\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request) and [reject\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request). A cancelled request will go back to the pending list while a rejected request will go to the rejected list. In both cases, the user will lose access to the repo.

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

[](#huggingface_hub.HfApi.list_accepted_access_requests.example)

Example:

Copied

\>>> from huggingface\_hub import list\_accepted\_access\_requests

\>>> requests = list\_accepted\_access\_requests("meta-llama/Llama-2-7b")
\>>> len(requests)
411
\>>> requests\[0\]
\[
    AccessRequest(
        username='clem',
        fullname='Clem 🤗',
        email='\*\*\*',
        timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
        status='accepted',
        fields=None,
    ),
    ...
\]

#### list\_collections

[](#huggingface_hub.HfApi.list_collections)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7801)

( owner: Union\[List\[str\], str, None\] = Noneitem: Union\[List\[str\], str, None\] = Nonesort: Optional\[Literal\['lastModified', 'trending', 'upvotes'\]\] = Nonelimit: Optional\[int\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[Collection]`

Parameters

*   [](#huggingface_hub.HfApi.list_collections.owner)**owner** (`List[str]` or `str`, _optional_) — Filter by owner’s username.
*   [](#huggingface_hub.HfApi.list_collections.item)**item** (`List[str]` or `str`, _optional_) — Filter collections containing a particular items. Example: `"models/teknium/OpenHermes-2.5-Mistral-7B"`, `"datasets/squad"` or `"papers/2311.12983"`.
*   [](#huggingface_hub.HfApi.list_collections.sort)**sort** (`Literal["lastModified", "trending", "upvotes"]`, _optional_) — Sort collections by last modified, trending or upvotes.
*   [](#huggingface_hub.HfApi.list_collections.limit)**limit** (`int`, _optional_) — Maximum number of collections to be returned.
*   [](#huggingface_hub.HfApi.list_collections.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[Collection]`

export const metadata = 'undefined';

an iterable of [Collection](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.Collection) objects.

List collections on the Huggingface Hub, given some filters.

When listing collections, the item list per collection is truncated to 4 items maximum. To retrieve all items from a collection, you must use [get\_collection()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.get_collection).

#### list\_datasets

[](#huggingface_hub.HfApi.list_datasets)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1958)

( filter: Union\[str, Iterable\[str\], None\] = Noneauthor: Optional\[str\] = Nonebenchmark: Optional\[Union\[str, List\[str\]\]\] = Nonedataset\_name: Optional\[str\] = Nonegated: Optional\[bool\] = Nonelanguage\_creators: Optional\[Union\[str, List\[str\]\]\] = Nonelanguage: Optional\[Union\[str, List\[str\]\]\] = Nonemultilinguality: Optional\[Union\[str, List\[str\]\]\] = Nonesize\_categories: Optional\[Union\[str, List\[str\]\]\] = Nonetags: Optional\[Union\[str, List\[str\]\]\] = Nonetask\_categories: Optional\[Union\[str, List\[str\]\]\] = Nonetask\_ids: Optional\[Union\[str, List\[str\]\]\] = Nonesearch: Optional\[str\] = Nonesort: Optional\[Union\[Literal\['last\_modified'\], str\]\] = Nonedirection: Optional\[Literal\[-1\]\] = Nonelimit: Optional\[int\] = Noneexpand: Optional\[List\[ExpandDatasetProperty\_T\]\] = Nonefull: Optional\[bool\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[DatasetInfo]`

Expand 19 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_datasets.filter)**filter** (`str` or `Iterable[str]`, _optional_) — A string or list of string to filter datasets on the hub.
*   [](#huggingface_hub.HfApi.list_datasets.author)**author** (`str`, _optional_) — A string which identify the author of the returned datasets.
*   [](#huggingface_hub.HfApi.list_datasets.benchmark)**benchmark** (`str` or `List`, _optional_) — A string or list of strings that can be used to identify datasets on the Hub by their official benchmark.
*   [](#huggingface_hub.HfApi.list_datasets.dataset_name)**dataset\_name** (`str`, _optional_) — A string or list of strings that can be used to identify datasets on the Hub by its name, such as `SQAC` or `wikineural`
*   [](#huggingface_hub.HfApi.list_datasets.gated)**gated** (`bool`, _optional_) — A boolean to filter datasets on the Hub that are gated or not. By default, all datasets are returned. If `gated=True` is passed, only gated datasets are returned. If `gated=False` is passed, only non-gated datasets are returned.
*   [](#huggingface_hub.HfApi.list_datasets.language_creators)**language\_creators** (`str` or `List`, _optional_) — A string or list of strings that can be used to identify datasets on the Hub with how the data was curated, such as `crowdsourced` or `machine_generated`.
*   [](#huggingface_hub.HfApi.list_datasets.language)**language** (`str` or `List`, _optional_) — A string or list of strings representing a two-character language to filter datasets by on the Hub.
*   [](#huggingface_hub.HfApi.list_datasets.multilinguality)**multilinguality** (`str` or `List`, _optional_) — A string or list of strings representing a filter for datasets that contain multiple languages.
*   [](#huggingface_hub.HfApi.list_datasets.size_categories)**size\_categories** (`str` or `List`, _optional_) — A string or list of strings that can be used to identify datasets on the Hub by the size of the dataset such as `100K<n<1M` or `1M<n<10M`.
*   [](#huggingface_hub.HfApi.list_datasets.tags)**tags** (`str` or `List`, _optional_) — A string tag or a list of tags to filter datasets on the Hub.
*   [](#huggingface_hub.HfApi.list_datasets.task_categories)**task\_categories** (`str` or `List`, _optional_) — A string or list of strings that can be used to identify datasets on the Hub by the designed task, such as `audio_classification` or `named_entity_recognition`.
*   [](#huggingface_hub.HfApi.list_datasets.task_ids)**task\_ids** (`str` or `List`, _optional_) — A string or list of strings that can be used to identify datasets on the Hub by the specific task such as `speech_emotion_recognition` or `paraphrase`.
*   [](#huggingface_hub.HfApi.list_datasets.search)**search** (`str`, _optional_) — A string that will be contained in the returned datasets.
*   [](#huggingface_hub.HfApi.list_datasets.sort)**sort** (`Literal["last_modified"]` or `str`, _optional_) — The key with which to sort the resulting models. Possible values are “last\_modified”, “trending\_score”, “created\_at”, “downloads” and “likes”.
*   [](#huggingface_hub.HfApi.list_datasets.direction)**direction** (`Literal[-1]` or `int`, _optional_) — Direction in which to sort. The value `-1` sorts by descending order while all other values sort by ascending order.
*   [](#huggingface_hub.HfApi.list_datasets.limit)**limit** (`int`, _optional_) — The limit on the number of datasets fetched. Leaving this option to `None` fetches all datasets.
*   [](#huggingface_hub.HfApi.list_datasets.expand)**expand** (`List[ExpandDatasetProperty_T]`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `full` is passed. Possible values are `"author"`, `"cardData"`, `"citation"`, `"createdAt"`, `"disabled"`, `"description"`, `"downloads"`, `"downloadsAllTime"`, `"gated"`, `"lastModified"`, `"likes"`, `"paperswithcode_id"`, `"private"`, `"siblings"`, `"sha"`, `"tags"`, `"trendingScore"`, `"usedStorage"` and `"resourceGroup"`.
*   [](#huggingface_hub.HfApi.list_datasets.full)**full** (`bool`, _optional_) — Whether to fetch all dataset data, including the `last_modified`, the `card_data` and the files. Can contain useful information such as the PapersWithCode ID.
*   [](#huggingface_hub.HfApi.list_datasets.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[DatasetInfo]`

export const metadata = 'undefined';

an iterable of [huggingface\_hub.hf\_api.DatasetInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.DatasetInfo) objects.

List datasets hosted on the Huggingface Hub, given some filters.

[](#huggingface_hub.HfApi.list_datasets.example)

Example usage with the `filter` argument:

Copied

\>>> from huggingface\_hub import HfApi

\>>> api = HfApi()

\# List all datasets
\>>> api.list\_datasets()

\# List only the text classification datasets
\>>> api.list\_datasets(filter\="task\_categories:text-classification")

\# List only the datasets in russian for language modeling
\>>> api.list\_datasets(
...     filter\=("language:ru", "task\_ids:language-modeling")
... )

\# List FiftyOne datasets (identified by the tag "fiftyone" in dataset card)
\>>> api.list\_datasets(tags="fiftyone")

[](#huggingface_hub.HfApi.list_datasets.example-2)

Example usage with the `search` argument:

Copied

\>>> from huggingface\_hub import HfApi

\>>> api = HfApi()

\# List all datasets with "text" in their name
\>>> api.list\_datasets(search="text")

\# List all datasets with "text" in their name made by google
\>>> api.list\_datasets(search="text", author="google")

#### list\_inference\_endpoints

[](#huggingface_hub.HfApi.list_inference_endpoints)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7264)

( namespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';List[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Parameters

*   [](#huggingface_hub.HfApi.list_inference_endpoints.namespace)**namespace** (`str`, _optional_) — The namespace to list endpoints for. Defaults to the current user. Set to `"*"` to list all endpoints from all namespaces (i.e. personal namespace and all orgs the user belongs to).
*   [](#huggingface_hub.HfApi.list_inference_endpoints.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

List[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

A list of all inference endpoints for the given namespace.

Lists all inference endpoints for the given namespace.

[](#huggingface_hub.HfApi.list_inference_endpoints.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> api.list\_inference\_endpoints()
\[InferenceEndpoint(name='my-endpoint', ...), ...\]

#### list\_liked\_repos

[](#huggingface_hub.HfApi.list_liked_repos)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2330)

( user: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[UserLikes](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.UserLikes)

Parameters

*   [](#huggingface_hub.HfApi.list_liked_repos.user)**user** (`str`, _optional_) — Name of the user for which you want to fetch the likes.
*   [](#huggingface_hub.HfApi.list_liked_repos.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[UserLikes](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.UserLikes)

export const metadata = 'undefined';

object containing the user name and 3 lists of repo ids (1 for models, 1 for datasets and 1 for Spaces).

Raises

export const metadata = 'undefined';

`ValueError`

export const metadata = 'undefined';

*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If `user` is not passed and no token found (either from argument or from machine).

List all public repos liked by a user on huggingface.co.

This list is public so token is optional. If `user` is not passed, it defaults to the logged in user.

See also [unlike()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.unlike).

[](#huggingface_hub.HfApi.list_liked_repos.example)

Example:

Copied

\>>> from huggingface\_hub import list\_liked\_repos

\>>> likes = list\_liked\_repos("julien-c")

\>>> likes.user
"julien-c"

\>>> likes.models
\["osanseviero/streamlit\_1.15", "Xhaheen/ChatGPT\_HF", ...\]

#### list\_models

[](#huggingface_hub.HfApi.list_models)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1738)

( filter: Union\[str, Iterable\[str\], None\] = Noneauthor: Optional\[str\] = Nonegated: Optional\[bool\] = Noneinference: Optional\[Literal\['cold', 'frozen', 'warm'\]\] = Nonelibrary: Optional\[Union\[str, List\[str\]\]\] = Nonelanguage: Optional\[Union\[str, List\[str\]\]\] = Nonemodel\_name: Optional\[str\] = Nonetask: Optional\[Union\[str, List\[str\]\]\] = Nonetrained\_dataset: Optional\[Union\[str, List\[str\]\]\] = Nonetags: Optional\[Union\[str, List\[str\]\]\] = Nonesearch: Optional\[str\] = Nonepipeline\_tag: Optional\[str\] = Noneemissions\_thresholds: Optional\[Tuple\[float, float\]\] = Nonesort: Union\[Literal\['last\_modified'\], str, None\] = Nonedirection: Optional\[Literal\[-1\]\] = Nonelimit: Optional\[int\] = Noneexpand: Optional\[List\[ExpandModelProperty\_T\]\] = Nonefull: Optional\[bool\] = NonecardData: bool = Falsefetch\_config: bool = Falsetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[ModelInfo]`

Expand 21 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_models.filter)**filter** (`str` or `Iterable[str]`, _optional_) — A string or list of string to filter models on the Hub.
*   [](#huggingface_hub.HfApi.list_models.author)**author** (`str`, _optional_) — A string which identify the author (user or organization) of the returned models.
*   [](#huggingface_hub.HfApi.list_models.gated)**gated** (`bool`, _optional_) — A boolean to filter models on the Hub that are gated or not. By default, all models are returned. If `gated=True` is passed, only gated models are returned. If `gated=False` is passed, only non-gated models are returned.
*   [](#huggingface_hub.HfApi.list_models.inference)**inference** (`Literal["cold", "frozen", "warm"]`, _optional_) — A string to filter models on the Hub by their state on the Inference API. Warm models are available for immediate use. Cold models will be loaded on first inference call. Frozen models are not available in Inference API.
*   [](#huggingface_hub.HfApi.list_models.library)**library** (`str` or `List`, _optional_) — A string or list of strings of foundational libraries models were originally trained from, such as pytorch, tensorflow, or allennlp.
*   [](#huggingface_hub.HfApi.list_models.language)**language** (`str` or `List`, _optional_) — A string or list of strings of languages, both by name and country code, such as “en” or “English”
*   [](#huggingface_hub.HfApi.list_models.model_name)**model\_name** (`str`, _optional_) — A string that contain complete or partial names for models on the Hub, such as “bert” or “bert-base-cased”
*   [](#huggingface_hub.HfApi.list_models.task)**task** (`str` or `List`, _optional_) — A string or list of strings of tasks models were designed for, such as: “fill-mask” or “automatic-speech-recognition”
*   [](#huggingface_hub.HfApi.list_models.trained_dataset)**trained\_dataset** (`str` or `List`, _optional_) — A string tag or a list of string tags of the trained dataset for a model on the Hub.
*   [](#huggingface_hub.HfApi.list_models.tags)**tags** (`str` or `List`, _optional_) — A string tag or a list of tags to filter models on the Hub by, such as `text-generation` or `spacy`.
*   [](#huggingface_hub.HfApi.list_models.search)**search** (`str`, _optional_) — A string that will be contained in the returned model ids.
*   [](#huggingface_hub.HfApi.list_models.pipeline_tag)**pipeline\_tag** (`str`, _optional_) — A string pipeline tag to filter models on the Hub by, such as `summarization`.
*   [](#huggingface_hub.HfApi.list_models.emissions_thresholds)**emissions\_thresholds** (`Tuple`, _optional_) — A tuple of two ints or floats representing a minimum and maximum carbon footprint to filter the resulting models with in grams.
*   [](#huggingface_hub.HfApi.list_models.sort)**sort** (`Literal["last_modified"]` or `str`, _optional_) — The key with which to sort the resulting models. Possible values are “last\_modified”, “trending\_score”, “created\_at”, “downloads” and “likes”.
*   [](#huggingface_hub.HfApi.list_models.direction)**direction** (`Literal[-1]` or `int`, _optional_) — Direction in which to sort. The value `-1` sorts by descending order while all other values sort by ascending order.
*   [](#huggingface_hub.HfApi.list_models.limit)**limit** (`int`, _optional_) — The limit on the number of models fetched. Leaving this option to `None` fetches all models.
*   [](#huggingface_hub.HfApi.list_models.expand)**expand** (`List[ExpandModelProperty_T]`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `full`, `cardData` or `fetch_config` are passed. Possible values are `"author"`, `"baseModels"`, `"cardData"`, `"childrenModelCount"`, `"config"`, `"createdAt"`, `"disabled"`, `"downloads"`, `"downloadsAllTime"`, `"gated"`, `"gguf"`, `"inference"`, `"inferenceProviderMapping"`, `"lastModified"`, `"library_name"`, `"likes"`, `"mask_token"`, `"model-index"`, `"pipeline_tag"`, `"private"`, `"safetensors"`, `"sha"`, `"siblings"`, `"spaces"`, `"tags"`, `"transformersInfo"`, `"trendingScore"`, `"widgetData"`, `"usedStorage"` and `"resourceGroup"`.
*   [](#huggingface_hub.HfApi.list_models.full)**full** (`bool`, _optional_) — Whether to fetch all model data, including the `last_modified`, the `sha`, the files and the `tags`. This is set to `True` by default when using a filter.
*   [](#huggingface_hub.HfApi.list_models.cardData)**cardData** (`bool`, _optional_) — Whether to grab the metadata for the model as well. Can contain useful information such as carbon emissions, metrics, and datasets trained on.
*   [](#huggingface_hub.HfApi.list_models.fetch_config)**fetch\_config** (`bool`, _optional_) — Whether to fetch the model configs as well. This is not included in `full` due to its size.
*   [](#huggingface_hub.HfApi.list_models.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[ModelInfo]`

export const metadata = 'undefined';

an iterable of [huggingface\_hub.hf\_api.ModelInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.ModelInfo) objects.

List models hosted on the Huggingface Hub, given some filters.

[](#huggingface_hub.HfApi.list_models.example)

Example usage with the `filter` argument:

Copied

\>>> from huggingface\_hub import HfApi

\>>> api = HfApi()

\# List all models
\>>> api.list\_models()

\# List only the text classification models
\>>> api.list\_models(filter\="text-classification")

\# List only models from the AllenNLP library
\>>> api.list\_models(filter\="allennlp")

[](#huggingface_hub.HfApi.list_models.example-2)

Example usage with the `search` argument:

Copied

\>>> from huggingface\_hub import HfApi

\>>> api = HfApi()

\# List all models with "bert" in their name
\>>> api.list\_models(search="bert")

\# List all models with "bert" in their name made by google
\>>> api.list\_models(search="bert", author="google")

#### list\_organization\_members

[](#huggingface_hub.HfApi.list_organization_members)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9282)

( organization: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[User]`

Parameters

*   [](#huggingface_hub.HfApi.list_organization_members.organization)**organization** (`str`) — Name of the organization to get the members of.
*   [](#huggingface_hub.HfApi.list_organization_members.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[User]`

export const metadata = 'undefined';

A list of [User](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.User) objects with the members of the organization.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 If the organization does not exist on the Hub.

List of members of an organization on the Hub.

#### list\_papers

[](#huggingface_hub.HfApi.list_papers)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9366)

( query: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[PaperInfo]`

Parameters

*   [](#huggingface_hub.HfApi.list_papers.query)**query** (`str`, _optional_) — A search query string to find papers. If provided, returns papers that match the query.
*   [](#huggingface_hub.HfApi.list_papers.token)**token** (Union\[bool, str, None\], _optional_) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[PaperInfo]`

export const metadata = 'undefined';

an iterable of `huggingface_hub.hf_api.PaperInfo` objects.

List daily papers on the Hugging Face Hub given a search query.

[](#huggingface_hub.HfApi.list_papers.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi

\>>> api = HfApi()

\# List all papers with "attention" in their title
\>>> api.list\_papers(query="attention")

#### list\_pending\_access\_requests

[](#huggingface_hub.HfApi.list_pending_access_requests)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8267)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`List[AccessRequest]`

Expand 3 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_pending_access_requests.repo_id)**repo\_id** (`str`) — The id of the repo to get access requests for.
*   [](#huggingface_hub.HfApi.list_pending_access_requests.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to get access requests for. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.list_pending_access_requests.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`List[AccessRequest]`

export const metadata = 'undefined';

A list of `AccessRequest` objects. Each time contains a `username`, `email`, `status` and `timestamp` attribute. If the gated repo has a custom form, the `fields` attribute will be populated with user’s answers.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.

Get pending access requests for a given gated repo.

A pending request means the user has requested access to the repo but the request has not been processed yet. If the approval mode is automatic, this list should be empty. Pending requests can be accepted or rejected using [accept\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request) and [reject\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request).

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

[](#huggingface_hub.HfApi.list_pending_access_requests.example)

Example:

Copied

\>>> from huggingface\_hub import list\_pending\_access\_requests, accept\_access\_request

\# List pending requests
\>>> requests = list\_pending\_access\_requests("meta-llama/Llama-2-7b")
\>>> len(requests)
411
\>>> requests\[0\]
\[
    AccessRequest(
        username='clem',
        fullname='Clem 🤗',
        email='\*\*\*',
        timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
        status='pending',
        fields=None,
    ),
    ...
\]

\# Accept Clem's request
\>>> accept\_access\_request("meta-llama/Llama-2-7b", "clem")

#### list\_rejected\_access\_requests

[](#huggingface_hub.HfApi.list_rejected_access_requests)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8393)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`List[AccessRequest]`

Expand 3 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_rejected_access_requests.repo_id)**repo\_id** (`str`) — The id of the repo to get access requests for.
*   [](#huggingface_hub.HfApi.list_rejected_access_requests.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to get access requests for. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.list_rejected_access_requests.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`List[AccessRequest]`

export const metadata = 'undefined';

A list of `AccessRequest` objects. Each time contains a `username`, `email`, `status` and `timestamp` attribute. If the gated repo has a custom form, the `fields` attribute will be populated with user’s answers.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.

Get rejected access requests for a given gated repo.

A rejected request means the user has requested access to the repo and the request has been explicitly rejected by a repo owner (either you or another user from your organization). The user cannot download any file of the repo. Rejected requests can be accepted or cancelled at any time using [accept\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request) and [cancel\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request). A cancelled request will go back to the pending list while an accepted request will go to the accepted list.

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

[](#huggingface_hub.HfApi.list_rejected_access_requests.example)

Example:

Copied

\>>> from huggingface\_hub import list\_rejected\_access\_requests

\>>> requests = list\_rejected\_access\_requests("meta-llama/Llama-2-7b")
\>>> len(requests)
411
\>>> requests\[0\]
\[
    AccessRequest(
        username='clem',
        fullname='Clem 🤗',
        email='\*\*\*',
        timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
        status='rejected',
        fields=None,
    ),
    ...
\]

#### list\_repo\_commits

[](#huggingface_hub.HfApi.list_repo_commits)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3133)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = Nonerevision: Optional\[str\] = Noneformatted: bool = False ) → export const metadata = 'undefined';List\[[GitCommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitCommitInfo)\]

Expand 5 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_repo_commits.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.list_repo_commits.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if listing commits from a dataset or a Space, `None` or `"model"` if listing from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.list_repo_commits.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.list_repo_commits.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.list_repo_commits.formatted)**formatted** (`bool`) — Whether to return the HTML-formatted title and description of the commits. Defaults to False.

Returns

export const metadata = 'undefined';

List\[[GitCommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitCommitInfo)\]

export const metadata = 'undefined';

list of objects containing information about the commits for a repo on the Hub.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If revision is not found (error 404) on the repo.

Get the list of commits of a given revision for a repo on the Hub.

Commits are sorted by date (last commit first).

[](#huggingface_hub.HfApi.list_repo_commits.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()

\# Commits are sorted by date (last commit first)
\>>> initial\_commit = api.list\_repo\_commits("gpt2")\[-1\]

\# Initial commit is always a system commit containing the \`.gitattributes\` file.
\>>> initial\_commit
GitCommitInfo(
    commit\_id='9b865efde13a30c13e0a33e536cf3e4a5a9d71d8',
    authors=\['system'\],
    created\_at=datetime.datetime(2019, 2, 18, 10, 36, 15, tzinfo=datetime.timezone.utc),
    title='initial commit',
    message='',
    formatted\_title=None,
    formatted\_message=None
)

\# Create an empty branch by deriving from initial commit
\>>> api.create\_branch("gpt2", "new\_empty\_branch", revision=initial\_commit.commit\_id)

#### list\_repo\_files

[](#huggingface_hub.HfApi.list_repo_files)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2892)

( repo\_id: strrevision: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None ) → export const metadata = 'undefined';`List[str]`

Parameters

*   [](#huggingface_hub.HfApi.list_repo_files.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.list_repo_files.revision)**revision** (`str`, _optional_) — The revision of the repository from which to get the information.
*   [](#huggingface_hub.HfApi.list_repo_files.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.list_repo_files.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`List[str]`

export const metadata = 'undefined';

the list of files in a given repository.

Get the list of files in a given repo.

#### list\_repo\_likers

[](#huggingface_hub.HfApi.list_repo_likers)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2406)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[User]`

Parameters

*   [](#huggingface_hub.HfApi.list_repo_likers.repo_id)**repo\_id** (`str`) — The repository to retrieve . Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.list_repo_likers.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.list_repo_likers.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.

Returns

export const metadata = 'undefined';

`Iterable[User]`

export const metadata = 'undefined';

an iterable of [huggingface\_hub.hf\_api.User](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.User) objects.

List all users who liked a given repo on the hugging Face Hub.

See also [list\_liked\_repos()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_liked_repos).

#### list\_repo\_refs

[](#huggingface_hub.HfApi.list_repo_refs)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3061)

( repo\_id: strrepo\_type: Optional\[str\] = Noneinclude\_pull\_requests: bool = Falsetoken: Union\[str, bool, None\] = None ) → export const metadata = 'undefined';[GitRefs](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitRefs)

Parameters

*   [](#huggingface_hub.HfApi.list_repo_refs.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.list_repo_refs.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if listing refs from a dataset or a Space, `None` or `"model"` if listing from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.list_repo_refs.include_pull_requests)**include\_pull\_requests** (`bool`, _optional_) — Whether to include refs from pull requests in the list. Defaults to `False`.
*   [](#huggingface_hub.HfApi.list_repo_refs.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[GitRefs](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitRefs)

export const metadata = 'undefined';

object containing all information about branches and tags for a repo on the Hub.

Get the list of refs of a given repo (both tags and branches).

[](#huggingface_hub.HfApi.list_repo_refs.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> api.list\_repo\_refs("gpt2")
GitRefs(branches=\[GitRefInfo(name='main', ref='refs/heads/main', target\_commit='e7da7f221d5bf496a48136c0cd264e630fe9fcc8')\], converts=\[\], tags=\[\])

\>>> api.list\_repo\_refs("bigcode/the-stack", repo\_type='dataset')
GitRefs(
    branches=\[
        GitRefInfo(name='main', ref='refs/heads/main', target\_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
        GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target\_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
    \],
    converts=\[\],
    tags=\[
        GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target\_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
    \]
)

#### list\_repo\_tree

[](#huggingface_hub.HfApi.list_repo_tree)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2929)

( repo\_id: strpath\_in\_repo: Optional\[str\] = Nonerecursive: bool = Falseexpand: bool = Falserevision: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None ) → export const metadata = 'undefined';`Iterable[Union[RepoFile, RepoFolder]]`

Expand 7 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_repo_tree.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.list_repo_tree.path_in_repo)**path\_in\_repo** (`str`, _optional_) — Relative path of the tree (folder) in the repo, for example: `"checkpoints/1fec34a/results"`. Will default to the root tree (folder) of the repository.
*   [](#huggingface_hub.HfApi.list_repo_tree.recursive)**recursive** (`bool`, _optional_, defaults to `False`) — Whether to list tree’s files and folders recursively.
*   [](#huggingface_hub.HfApi.list_repo_tree.expand)**expand** (`bool`, _optional_, defaults to `False`) — Whether to fetch more information about the tree’s files and folders (e.g. last commit and files’ security scan results). This operation is more expensive for the server so only 50 results are returned per page (instead of 1000). As pagination is implemented in `huggingface_hub`, this is transparent for you except for the time it takes to get the results.
*   [](#huggingface_hub.HfApi.list_repo_tree.revision)**revision** (`str`, _optional_) — The revision of the repository from which to get the tree. Defaults to `"main"` branch.
*   [](#huggingface_hub.HfApi.list_repo_tree.repo_type)**repo\_type** (`str`, _optional_) — The type of the repository from which to get the tree (`"model"`, `"dataset"` or `"space"`. Defaults to `"model"`.
*   [](#huggingface_hub.HfApi.list_repo_tree.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[Union[RepoFile, RepoFolder]]`

export const metadata = 'undefined';

The information about the tree’s files and folders, as an iterable of `RepoFile` and `RepoFolder` objects. The order of the files and folders is not guaranteed.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) or [EntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If revision is not found (error 404) on the repo.
*   [EntryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError) — If the tree (folder) does not exist (error 404) on the repo.

List a repo tree’s files and folders and get information about them.

Examples:

Get information about a repo’s tree.

[](#huggingface_hub.HfApi.list_repo_tree.example)

Copied

\>>> from huggingface\_hub import list\_repo\_tree
\>>> repo\_tree = list\_repo\_tree("lysandre/arxiv-nlp")
\>>> repo\_tree
<generator object HfApi.list\_repo\_tree at 0x7fa4088e1ac0\>
\>>> list(repo\_tree)
\[
    RepoFile(path='.gitattributes', size=391, blob\_id='ae8c63daedbd4206d7d40126955d4e6ab1c80f8f', lfs=None, last\_commit=None, security=None),
    RepoFile(path='README.md', size=391, blob\_id='43bd404b159de6fba7c2f4d3264347668d43af25', lfs=None, last\_commit=None, security=None),
    RepoFile(path='config.json', size=554, blob\_id='2f9618c3a19b9a61add74f70bfb121335aeef666', lfs=None, last\_commit=None, security=None),
    RepoFile(
        path='flax\_model.msgpack', size=497764107, blob\_id='8095a62ccb4d806da7666fcda07467e2d150218e',
        lfs={'size': 497764107, 'sha256': 'd88b0d6a6ff9c3f8151f9d3228f57092aaea997f09af009eefd7373a77b5abb9', 'pointer\_size': 134}, last\_commit=None, security=None
    ),
    RepoFile(path='merges.txt', size=456318, blob\_id='226b0752cac7789c48f0cb3ec53eda48b7be36cc', lfs=None, last\_commit=None, security=None),
    RepoFile(
        path='pytorch\_model.bin', size=548123560, blob\_id='64eaa9c526867e404b68f2c5d66fd78e27026523',
        lfs={'size': 548123560, 'sha256': '9be78edb5b928eba33aa88f431551348f7466ba9f5ef3daf1d552398722a5436', 'pointer\_size': 134}, last\_commit=None, security=None
    ),
    RepoFile(path='vocab.json', size=898669, blob\_id='b00361fece0387ca34b4b8b8539ed830d644dbeb', lfs=None, last\_commit=None, security=None)\]
\]

Get even more information about a repo’s tree (last commit and files’ security scan results)

[](#huggingface_hub.HfApi.list_repo_tree.example-2)

Copied

\>>> from huggingface\_hub import list\_repo\_tree
\>>> repo\_tree = list\_repo\_tree("prompthero/openjourney-v4", expand=True)
\>>> list(repo\_tree)
\[
    RepoFolder(
        path='feature\_extractor',
        tree\_id='aa536c4ea18073388b5b0bc791057a7296a00398',
        last\_commit={
            'oid': '47b62b20b20e06b9de610e840282b7e6c3d51190',
            'title': 'Upload diffusers weights (#48)',
            'date': datetime.datetime(2023, 3, 21, 9, 5, 27, tzinfo=datetime.timezone.utc)
        }
    ),
    RepoFolder(
        path='safety\_checker',
        tree\_id='65aef9d787e5557373fdf714d6c34d4fcdd70440',
        last\_commit={
            'oid': '47b62b20b20e06b9de610e840282b7e6c3d51190',
            'title': 'Upload diffusers weights (#48)',
            'date': datetime.datetime(2023, 3, 21, 9, 5, 27, tzinfo=datetime.timezone.utc)
        }
    ),
    RepoFile(
        path='model\_index.json',
        size=582,
        blob\_id='d3d7c1e8c3e78eeb1640b8e2041ee256e24c9ee1',
        lfs=None,
        last\_commit={
            'oid': 'b195ed2d503f3eb29637050a886d77bd81d35f0e',
            'title': 'Fix deprecation warning by changing \`CLIPFeatureExtractor\` to \`CLIPImageProcessor\`. (#54)',
            'date': datetime.datetime(2023, 5, 15, 21, 41, 59, tzinfo=datetime.timezone.utc)
        },
        security={
            'safe': True,
            'av\_scan': {'virusFound': False, 'virusNames': None},
            'pickle\_import\_scan': None
        }
    )
    ...
\]

#### list\_spaces

[](#huggingface_hub.HfApi.list_spaces)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2169)

( filter: Union\[str, Iterable\[str\], None\] = Noneauthor: Optional\[str\] = Nonesearch: Optional\[str\] = Nonedatasets: Union\[str, Iterable\[str\], None\] = Nonemodels: Union\[str, Iterable\[str\], None\] = Nonelinked: bool = Falsesort: Union\[Literal\['last\_modified'\], str, None\] = Nonedirection: Optional\[Literal\[-1\]\] = Nonelimit: Optional\[int\] = Noneexpand: Optional\[List\[ExpandSpaceProperty\_T\]\] = Nonefull: Optional\[bool\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[SpaceInfo]`

Expand 12 parameters

Parameters

*   [](#huggingface_hub.HfApi.list_spaces.filter)**filter** (`str` or `Iterable`, _optional_) — A string tag or list of tags that can be used to identify Spaces on the Hub.
*   [](#huggingface_hub.HfApi.list_spaces.author)**author** (`str`, _optional_) — A string which identify the author of the returned Spaces.
*   [](#huggingface_hub.HfApi.list_spaces.search)**search** (`str`, _optional_) — A string that will be contained in the returned Spaces.
*   [](#huggingface_hub.HfApi.list_spaces.datasets)**datasets** (`str` or `Iterable`, _optional_) — Whether to return Spaces that make use of a dataset. The name of a specific dataset can be passed as a string.
*   [](#huggingface_hub.HfApi.list_spaces.models)**models** (`str` or `Iterable`, _optional_) — Whether to return Spaces that make use of a model. The name of a specific model can be passed as a string.
*   [](#huggingface_hub.HfApi.list_spaces.linked)**linked** (`bool`, _optional_) — Whether to return Spaces that make use of either a model or a dataset.
*   [](#huggingface_hub.HfApi.list_spaces.sort)**sort** (`Literal["last_modified"]` or `str`, _optional_) — The key with which to sort the resulting models. Possible values are “last\_modified”, “trending\_score”, “created\_at” and “likes”.
*   [](#huggingface_hub.HfApi.list_spaces.direction)**direction** (`Literal[-1]` or `int`, _optional_) — Direction in which to sort. The value `-1` sorts by descending order while all other values sort by ascending order.
*   [](#huggingface_hub.HfApi.list_spaces.limit)**limit** (`int`, _optional_) — The limit on the number of Spaces fetched. Leaving this option to `None` fetches all Spaces.
*   [](#huggingface_hub.HfApi.list_spaces.expand)**expand** (`List[ExpandSpaceProperty_T]`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `full` is passed. Possible values are `"author"`, `"cardData"`, `"datasets"`, `"disabled"`, `"lastModified"`, `"createdAt"`, `"likes"`, `"models"`, `"private"`, `"runtime"`, `"sdk"`, `"siblings"`, `"sha"`, `"subdomain"`, `"tags"`, `"trendingScore"`, `"usedStorage"` and `"resourceGroup"`.
*   [](#huggingface_hub.HfApi.list_spaces.full)**full** (`bool`, _optional_) — Whether to fetch all Spaces data, including the `last_modified`, `siblings` and `card_data` fields.
*   [](#huggingface_hub.HfApi.list_spaces.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[SpaceInfo]`

export const metadata = 'undefined';

an iterable of [huggingface\_hub.hf\_api.SpaceInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.SpaceInfo) objects.

List spaces hosted on the Huggingface Hub, given some filters.

#### list\_user\_followers

[](#huggingface_hub.HfApi.list_user_followers)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9310)

( username: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[User]`

Parameters

*   [](#huggingface_hub.HfApi.list_user_followers.username)**username** (`str`) — Username of the user to get the followers of.
*   [](#huggingface_hub.HfApi.list_user_followers.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[User]`

export const metadata = 'undefined';

A list of [User](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.User) objects with the followers of the user.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 If the user does not exist on the Hub.

Get the list of followers of a user on the Hub.

#### list\_user\_following

[](#huggingface_hub.HfApi.list_user_following)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9338)

( username: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Iterable[User]`

Parameters

*   [](#huggingface_hub.HfApi.list_user_following.username)**username** (`str`) — Username of the user to get the users followed by.
*   [](#huggingface_hub.HfApi.list_user_following.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Iterable[User]`

export const metadata = 'undefined';

A list of [User](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.User) objects with the users followed by the user.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 If the user does not exist on the Hub.

Get the list of users followed by a user on the Hub.

#### list\_webhooks

[](#huggingface_hub.HfApi.list_webhooks)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8753)

( token: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`List[WebhookInfo]`

Parameters

*   [](#huggingface_hub.HfApi.list_webhooks.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`List[WebhookInfo]`

export const metadata = 'undefined';

List of webhook info objects.

List all configured webhooks.

[](#huggingface_hub.HfApi.list_webhooks.example)

Example:

Copied

\>>> from huggingface\_hub import list\_webhooks
\>>> webhooks = list\_webhooks()
\>>> len(webhooks)
2
\>>> webhooks\[0\]
WebhookInfo(
    id\="654bbbc16f2ec14d77f109cc",
    watched=\[WebhookWatchedItem(type\="user", name="julien-c"), WebhookWatchedItem(type\="org", name="HuggingFaceH4")\],
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    secret="my-secret",
    domains=\["repo", "discussion"\],
    disabled=False,
)

#### merge\_pull\_request

[](#huggingface_hub.HfApi.merge_pull_request)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6525)

( repo\_id: strdiscussion\_num: inttoken: Union\[bool, str, None\] = Nonecomment: Optional\[str\] = Nonerepo\_type: Optional\[str\] = None ) → export const metadata = 'undefined';[DiscussionStatusChange](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionStatusChange)

Parameters

*   [](#huggingface_hub.HfApi.merge_pull_request.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.merge_pull_request.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.merge_pull_request.comment)**comment** (`str`, _optional_) — An optional comment to post with the status change.
*   [](#huggingface_hub.HfApi.merge_pull_request.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.merge_pull_request.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[DiscussionStatusChange](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionStatusChange)

export const metadata = 'undefined';

the status change event

Merges a Pull Request.

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### model\_info

[](#huggingface_hub.HfApi.model_info)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2445)

( repo\_id: strrevision: Optional\[str\] = Nonetimeout: Optional\[float\] = NonesecurityStatus: Optional\[bool\] = Nonefiles\_metadata: bool = Falseexpand: Optional\[List\[ExpandModelProperty\_T\]\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[huggingface\_hub.hf\_api.ModelInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.ModelInfo)

Expand 7 parameters

Parameters

*   [](#huggingface_hub.HfApi.model_info.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.model_info.revision)**revision** (`str`, _optional_) — The revision of the model repository from which to get the information.
*   [](#huggingface_hub.HfApi.model_info.timeout)**timeout** (`float`, _optional_) — Whether to set a timeout for the request to the Hub.
*   [](#huggingface_hub.HfApi.model_info.securityStatus)**securityStatus** (`bool`, _optional_) — Whether to retrieve the security status from the model repository as well. The security status will be returned in the `security_repo_status` field.
*   [](#huggingface_hub.HfApi.model_info.files_metadata)**files\_metadata** (`bool`, _optional_) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to `False`.
*   [](#huggingface_hub.HfApi.model_info.expand)**expand** (`List[ExpandModelProperty_T]`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `securityStatus` or `files_metadata` are passed. Possible values are `"author"`, `"baseModels"`, `"cardData"`, `"childrenModelCount"`, `"config"`, `"createdAt"`, `"disabled"`, `"downloads"`, `"downloadsAllTime"`, `"gated"`, `"gguf"`, `"inference"`, `"inferenceProviderMapping"`, `"lastModified"`, `"library_name"`, `"likes"`, `"mask_token"`, `"model-index"`, `"pipeline_tag"`, `"private"`, `"safetensors"`, `"sha"`, `"siblings"`, `"spaces"`, `"tags"`, `"transformersInfo"`, `"trendingScore"`, `"widgetData"`, `"usedStorage"` and `"resourceGroup"`.
*   [](#huggingface_hub.HfApi.model_info.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[huggingface\_hub.hf\_api.ModelInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.ModelInfo)

export const metadata = 'undefined';

The model repository information.

Get info on one specific model on huggingface.co

Model can be private if you pass an acceptable token or are logged in.

Raises the following errors:

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) If the revision to download from cannot be found.

#### move\_repo

[](#huggingface_hub.HfApi.move_repo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3713)

( from\_id: strto\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.move_repo.from_id)**from\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`. Original repository identifier.
*   [](#huggingface_hub.HfApi.move_repo.to_id)**to\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`. Final repository identifier.
*   [](#huggingface_hub.HfApi.move_repo.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.move_repo.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Moving a repository from namespace1/repo\_name1 to namespace2/repo\_name2

Note there are certain limitations. For more information about moving repositories, please see [https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo).

Raises the following errors:

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### paper\_info

[](#huggingface_hub.HfApi.paper_info)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L9412)

( id: str ) → export const metadata = 'undefined';`PaperInfo`

Parameters

*   [](#huggingface_hub.HfApi.paper_info.id)**id** (`str`, **optional**) — ArXiv id of the paper.

Returns

export const metadata = 'undefined';

`PaperInfo`

export const metadata = 'undefined';

A `PaperInfo` object.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 If the paper does not exist on the Hub.

Get information for a paper on the Hub.

#### parse\_safetensors\_file\_metadata

[](#huggingface_hub.HfApi.parse_safetensors_file_metadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5536)

( repo\_id: strfilename: strrepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`SafetensorsFileMetadata`

Expand 5 parameters

Parameters

*   [](#huggingface_hub.HfApi.parse_safetensors_file_metadata.repo_id)**repo\_id** (`str`) — A user or an organization name and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.parse_safetensors_file_metadata.filename)**filename** (`str`) — The name of the file in the repo.
*   [](#huggingface_hub.HfApi.parse_safetensors_file_metadata.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if the file is in a dataset or space, `None` or `"model"` if in a model. Default is `None`.
*   [](#huggingface_hub.HfApi.parse_safetensors_file_metadata.revision)**revision** (`str`, _optional_) — The git revision to fetch the file from. Can be a branch name, a tag, or a commit hash. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.parse_safetensors_file_metadata.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`SafetensorsFileMetadata`

export const metadata = 'undefined';

information related to a safetensors file.

Raises

export const metadata = 'undefined';

`NotASafetensorsRepoError` or `SafetensorsParsingError`

export const metadata = 'undefined';

*   `NotASafetensorsRepoError` — If the repo is not a safetensors repo i.e. doesn’t have either a `model.safetensors` or a `model.safetensors.index.json` file.
*   `SafetensorsParsingError` — If a safetensors file header couldn’t be parsed correctly.

Parse metadata from a safetensors file on the Hub.

To parse metadata from all safetensors files in a repo at once, use [get\_safetensors\_metadata()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.get_safetensors_metadata).

For more details regarding the safetensors format, check out [https://huggingface.co/docs/safetensors/index#format](https://huggingface.co/docs/safetensors/index#format).

#### pause\_inference\_endpoint

[](#huggingface_hub.HfApi.pause_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7671)

( name: strnamespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Parameters

*   [](#huggingface_hub.HfApi.pause_inference_endpoint.name)**name** (`str`) — The name of the Inference Endpoint to pause.
*   [](#huggingface_hub.HfApi.pause_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace in which the Inference Endpoint is located. Defaults to the current user.
*   [](#huggingface_hub.HfApi.pause_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

information about the paused Inference Endpoint.

Pause an Inference Endpoint.

A paused Inference Endpoint will not be charged. It can be resumed at any time using [resume\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint). This is different than scaling the Inference Endpoint to zero with [scale\_to\_zero\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint), which would be automatically restarted when a request is made to it.

For convenience, you can also pause an Inference Endpoint using [pause\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint).

#### pause\_space

[](#huggingface_hub.HfApi.pause_space)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6987)

( repo\_id: strtoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Expand 2 parameters

Parameters

*   [](#huggingface_hub.HfApi.pause_space.repo_id)**repo\_id** (`str`) — ID of the Space to pause. Example: `"Salesforce/BLIP2"`.
*   [](#huggingface_hub.HfApi.pause_space.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about your Space including `stage=PAUSED` and requested hardware.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) or [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If your Space is not found (error 404). Most probably wrong repo\_id or your space is private but you are not authenticated.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — 403 Forbidden: only the owner of a Space can pause it. If you want to manage a Space that you don’t own, either ask the owner by opening a Discussion or duplicate the Space.
*   [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) — If your Space is a static Space. Static Spaces are always running and never billed. If you want to hide a static Space, you can set it to private.

Pause your Space.

A paused Space stops executing until manually restarted by its owner. This is different from the sleeping state in which free Spaces go after 48h of inactivity. Paused time is not billed to your account, no matter the hardware you’ve selected. To restart your Space, use [restart\_space()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.restart_space) and go to your Space settings page.

For more details, please visit [the docs](https://huggingface.co/docs/hub/spaces-gpus#pause).

#### preupload\_lfs\_files

[](#huggingface_hub.HfApi.preupload_lfs_files)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4108)

( repo\_id: stradditions: Iterable\[CommitOperationAdd\]token: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Nonenum\_threads: int = 5free\_memory: bool = Truegitignore\_content: Optional\[str\] = None )

Expand 8 parameters

Parameters

*   [](#huggingface_hub.HfApi.preupload_lfs_files.repo_id)**repo\_id** (`str`) — The repository in which you will commit the files, for example: `"username/custom_transformers"`.
*   [](#huggingface_hub.HfApi.preupload_lfs_files.operations)**operations** (`Iterable` of [CommitOperationAdd](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd)) — The list of files to upload. Warning: the objects in this list will be mutated to include information relative to the upload. Do not reuse the same objects for multiple commits.
*   [](#huggingface_hub.HfApi.preupload_lfs_files.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.preupload_lfs_files.repo_type)**repo\_type** (`str`, _optional_) — The type of repository to upload to (e.g. `"model"` -default-, `"dataset"` or `"space"`).
*   [](#huggingface_hub.HfApi.preupload_lfs_files.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.preupload_lfs_files.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not you plan to create a Pull Request with that commit. Defaults to `False`.
*   [](#huggingface_hub.HfApi.preupload_lfs_files.num_threads)**num\_threads** (`int`, _optional_) — Number of concurrent threads for uploading files. Defaults to 5. Setting it to 2 means at most 2 files will be uploaded concurrently.
*   [](#huggingface_hub.HfApi.preupload_lfs_files.gitignore_content)**gitignore\_content** (`str`, _optional_) — The content of the `.gitignore` file to know which files should be ignored. The order of priority is to first check if `gitignore_content` is passed, then check if the `.gitignore` file is present in the list of files to commit and finally default to the `.gitignore` file already hosted on the Hub (if any).

Pre-upload LFS files to S3 in preparation on a future commit.

This method is useful if you are generating the files to upload on-the-fly and you don’t want to store them in memory before uploading them all at once.

This is a power-user method. You shouldn’t need to call it directly to make a normal commit. Use [create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) directly instead.

Commit operations will be mutated during the process. In particular, the attached `path_or_fileobj` will be removed after the upload to save memory (and replaced by an empty `bytes` object). Do not reuse the same objects except to pass them to [create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit). If you don’t want to remove the attached content from the commit operation object, pass `free_memory=False`.

[](#huggingface_hub.HfApi.preupload_lfs_files.example)

Example:

Copied

\>>> from huggingface\_hub import CommitOperationAdd, preupload\_lfs\_files, create\_commit, create\_repo

\>>> repo\_id = create\_repo("test\_preupload").repo\_id

\# Generate and preupload LFS files one by one
\>>> operations = \[\] \# List of all \`CommitOperationAdd\` objects that will be generated
\>>> for i in range(5):
...     content = ... \# generate binary content
...     addition = CommitOperationAdd(path\_in\_repo=f"shard\_{i}\_of\_5.bin", path\_or\_fileobj=content)
...     preupload\_lfs\_files(repo\_id, additions=\[addition\]) \# upload + free memory
...     operations.append(addition)

\# Create commit
\>>> create\_commit(repo\_id, operations=operations, commit\_message="Commit all shards")

#### reject\_access\_request

[](#huggingface_hub.HfApi.reject_access_request)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8566)

( repo\_id: struser: strrepo\_type: Optional\[str\] = Nonerejection\_reason: Optional\[str\]token: Union\[bool, str, None\] = None )

Expand 5 parameters

Parameters

*   [](#huggingface_hub.HfApi.reject_access_request.repo_id)**repo\_id** (`str`) — The id of the repo to reject access request for.
*   [](#huggingface_hub.HfApi.reject_access_request.user)**user** (`str`) — The username of the user which access request should be rejected.
*   [](#huggingface_hub.HfApi.reject_access_request.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to reject access request for. Must be one of `model`, `dataset` or `space`. Defaults to `model`.
*   [](#huggingface_hub.HfApi.reject_access_request.rejection_reason)**rejection\_reason** (`str`, _optional_) — Optional rejection reason that will be visible to the user (max 200 characters).
*   [](#huggingface_hub.HfApi.reject_access_request.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Raises

export const metadata = 'undefined';

`HTTPError`

export const metadata = 'undefined';

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 400 if the repo is not gated.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 403 if you only have read-only access to the repo. This can be the case if you don’t have `write` or `admin` role in the organization the repo belongs to or if you passed a `read` token.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user does not exist on the Hub.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user access request cannot be found.
*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) — HTTP 404 if the user access request is already in the rejected list.

Reject an access request from a user for a given gated repo.

A rejected request will go to the rejected list. The user cannot download any file of the repo. Rejected requests can be accepted or cancelled at any time using [accept\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request) and [cancel\_access\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request). A cancelled request will go back to the pending list while an accepted request will go to the accepted list.

For more info about gated repos, see [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated).

#### rename\_discussion

[](#huggingface_hub.HfApi.rename_discussion)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6377)

( repo\_id: strdiscussion\_num: intnew\_title: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None ) → export const metadata = 'undefined';[DiscussionTitleChange](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionTitleChange)

Parameters

*   [](#huggingface_hub.HfApi.rename_discussion.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.rename_discussion.discussion_num)**discussion\_num** (`int`) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
*   [](#huggingface_hub.HfApi.rename_discussion.new_title)**new\_title** (`str`) — The new title for the discussion
*   [](#huggingface_hub.HfApi.rename_discussion.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.rename_discussion.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[DiscussionTitleChange](/docs/huggingface_hub/v0.29.2/en/package_reference/community#huggingface_hub.DiscussionTitleChange)

export const metadata = 'undefined';

the title change event

Renames a Discussion.

[](#huggingface_hub.HfApi.rename_discussion.example)

Examples:

Copied

\>>> new\_title = "New title, fixing a typo"
\>>> HfApi().rename\_discussion(
...     repo\_id="username/repo\_name",
...     discussion\_num=34
...     new\_title=new\_title
... )
\# DiscussionTitleChange(id='deadbeef0000000', type='title-change', ...)

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### repo\_exists

[](#huggingface_hub.HfApi.repo_exists)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2743)

( repo\_id: strrepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.repo_exists.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.repo_exists.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space, `None` or `"model"` if getting repository info from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.repo_exists.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Checks if a repository exists on the Hugging Face Hub.

[](#huggingface_hub.HfApi.repo_exists.example)

Examples:

Copied

\>>> from huggingface\_hub import repo\_exists
\>>> repo\_exists("google/gemma-7b")
True
\>>> repo\_exists("google/not-a-repo")
False

#### repo\_info

[](#huggingface_hub.HfApi.repo_info)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2669)

( repo\_id: strrevision: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonetimeout: Optional\[float\] = Nonefiles\_metadata: bool = Falseexpand: Optional\[Union\[ExpandModelProperty\_T, ExpandDatasetProperty\_T, ExpandSpaceProperty\_T\]\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';`Union[SpaceInfo, DatasetInfo, ModelInfo]`

Expand 7 parameters

Parameters

*   [](#huggingface_hub.HfApi.repo_info.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.repo_info.revision)**revision** (`str`, _optional_) — The revision of the repository from which to get the information.
*   [](#huggingface_hub.HfApi.repo_info.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space, `None` or `"model"` if getting repository info from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.repo_info.timeout)**timeout** (`float`, _optional_) — Whether to set a timeout for the request to the Hub.
*   [](#huggingface_hub.HfApi.repo_info.expand)**expand** (`ExpandModelProperty_T` or `ExpandDatasetProperty_T` or `ExpandSpaceProperty_T`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `files_metadata` is passed. For an exhaustive list of available properties, check out [model\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.model_info), [dataset\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info) or [space\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.space_info).
*   [](#huggingface_hub.HfApi.repo_info.files_metadata)**files\_metadata** (`bool`, _optional_) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to `False`.
*   [](#huggingface_hub.HfApi.repo_info.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

`Union[SpaceInfo, DatasetInfo, ModelInfo]`

export const metadata = 'undefined';

The repository information, as a [huggingface\_hub.hf\_api.DatasetInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.DatasetInfo), [huggingface\_hub.hf\_api.ModelInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.ModelInfo) or [huggingface\_hub.hf\_api.SpaceInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.SpaceInfo) object.

Get the info object for a given repo of a given type.

Raises the following errors:

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) If the revision to download from cannot be found.

#### request\_space\_hardware

[](#huggingface_hub.HfApi.request_space_hardware)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6883)

( repo\_id: strhardware: SpaceHardwaretoken: Union\[bool, str, None\] = Nonesleep\_time: Optional\[int\] = None ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Parameters

*   [](#huggingface_hub.HfApi.request_space_hardware.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.request_space_hardware.hardware)**hardware** (`str` or [SpaceHardware](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceHardware)) — Hardware on which to run the Space. Example: `"t4-medium"`.
*   [](#huggingface_hub.HfApi.request_space_hardware.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.request_space_hardware.sleep_time)**sleep\_time** (`int`, _optional_) — Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don’t want your Space to sleep (default behavior for upgraded hardware). For free hardware, you can’t configure the sleep time (value is fixed to 48 hours of inactivity). See [https://huggingface.co/docs/hub/spaces-gpus#sleep-time](https://huggingface.co/docs/hub/spaces-gpus#sleep-time) for more details.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about a Space including Space stage and hardware.

Request new hardware for a Space.

It is also possible to request hardware directly when creating the Space repo! See [create\_repo()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) for details.

#### request\_space\_storage

[](#huggingface_hub.HfApi.request_space_storage)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7190)

( repo\_id: strstorage: SpaceStoragetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Parameters

*   [](#huggingface_hub.HfApi.request_space_storage.repo_id)**repo\_id** (`str`) — ID of the Space to update. Example: `"open-llm-leaderboard/open_llm_leaderboard"`.
*   [](#huggingface_hub.HfApi.request_space_storage.storage)**storage** (`str` or [SpaceStorage](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceStorage)) — Storage tier. Either ‘small’, ‘medium’, or ‘large’.
*   [](#huggingface_hub.HfApi.request_space_storage.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about a Space including Space stage and hardware.

Request persistent storage for a Space.

It is not possible to decrease persistent storage after its granted. To do so, you must delete it via [delete\_space\_storage()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.delete_space_storage).

#### restart\_space

[](#huggingface_hub.HfApi.restart_space)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7026)

( repo\_id: strtoken: Union\[bool, str, None\] = Nonefactory\_reboot: bool = False ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Expand 3 parameters

Parameters

*   [](#huggingface_hub.HfApi.restart_space.repo_id)**repo\_id** (`str`) — ID of the Space to restart. Example: `"Salesforce/BLIP2"`.
*   [](#huggingface_hub.HfApi.restart_space.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.restart_space.factory_reboot)**factory\_reboot** (`bool`, _optional_) — If `True`, the Space will be rebuilt from scratch without caching any requirements.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about your Space.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) or [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If your Space is not found (error 404). Most probably wrong repo\_id or your space is private but you are not authenticated.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — 403 Forbidden: only the owner of a Space can restart it. If you want to restart a Space that you don’t own, either ask the owner by opening a Discussion or duplicate the Space.
*   [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) — If your Space is a static Space. Static Spaces are always running and never billed. If you want to hide a static Space, you can set it to private.

Restart your Space.

This is the only way to programmatically restart a Space if you’ve put it on Pause (see [pause\_space()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.pause_space)). You must be the owner of the Space to restart it. If you are using an upgraded hardware, your account will be billed as soon as the Space is restarted. You can trigger a restart no matter the current state of a Space.

For more details, please visit [the docs](https://huggingface.co/docs/hub/spaces-gpus#pause).

#### resume\_inference\_endpoint

[](#huggingface_hub.HfApi.resume_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7706)

( name: strnamespace: Optional\[str\] = Nonerunning\_ok: bool = Truetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Parameters

*   [](#huggingface_hub.HfApi.resume_inference_endpoint.name)**name** (`str`) — The name of the Inference Endpoint to resume.
*   [](#huggingface_hub.HfApi.resume_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace in which the Inference Endpoint is located. Defaults to the current user.
*   [](#huggingface_hub.HfApi.resume_inference_endpoint.running_ok)**running\_ok** (`bool`, _optional_) — If `True`, the method will not raise an error if the Inference Endpoint is already running. Defaults to `True`.
*   [](#huggingface_hub.HfApi.resume_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

information about the resumed Inference Endpoint.

Resume an Inference Endpoint.

For convenience, you can also resume an Inference Endpoint using [InferenceEndpoint.resume()](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.resume).

#### revision\_exists

[](#huggingface_hub.HfApi.revision_exists)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2787)

( repo\_id: strrevision: strrepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.revision_exists.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.revision_exists.revision)**revision** (`str`) — The revision of the repository to check.
*   [](#huggingface_hub.HfApi.revision_exists.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space, `None` or `"model"` if getting repository info from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.revision_exists.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Checks if a specific revision exists on a repo on the Hugging Face Hub.

[](#huggingface_hub.HfApi.revision_exists.example)

Examples:

Copied

\>>> from huggingface\_hub import revision\_exists
\>>> revision\_exists("google/gemma-7b", "float16")
True
\>>> revision\_exists("google/gemma-7b", "not-a-revision")
False

#### run\_as\_future

[](#huggingface_hub.HfApi.run_as_future)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1605)

( fn: Callable\[..., R\]\*args\*\*kwargs ) → export const metadata = 'undefined';`Future`

Parameters

*   [](#huggingface_hub.HfApi.run_as_future.fn)**fn** (`Callable`) — The method to run in the background.
*   [](#huggingface_hub.HfApi.run_as_future.*args,)**\*args,** \*\*kwargs — Arguments with which the method will be called.

Returns

export const metadata = 'undefined';

`Future`

export const metadata = 'undefined';

a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) instance to get the result of the task.

Run a method in the background and return a Future instance.

The main goal is to run methods without blocking the main thread (e.g. to push data during a training). Background jobs are queued to preserve order but are not ran in parallel. If you need to speed-up your scripts by parallelizing lots of call to the API, you must setup and use your own [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor).

Note: Most-used methods like [upload\_file()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file), [upload\_folder()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) and [create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) have a `run_as_future: bool` argument to directly call them in the background. This is equivalent to calling `api.run_as_future(...)` on them but less verbose.

[](#huggingface_hub.HfApi.run_as_future.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> future = api.run\_as\_future(api.whoami) \# instant
\>>> future.done()
False
\>>> future.result() \# wait until complete and return result
(...)
\>>> future.done()
True

#### scale\_to\_zero\_inference\_endpoint

[](#huggingface_hub.HfApi.scale_to_zero_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7752)

( name: strnamespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Parameters

*   [](#huggingface_hub.HfApi.scale_to_zero_inference_endpoint.name)**name** (`str`) — The name of the Inference Endpoint to scale to zero.
*   [](#huggingface_hub.HfApi.scale_to_zero_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace in which the Inference Endpoint is located. Defaults to the current user.
*   [](#huggingface_hub.HfApi.scale_to_zero_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

information about the scaled-to-zero Inference Endpoint.

Scale Inference Endpoint to zero.

An Inference Endpoint scaled to zero will not be charged. It will be resume on the next request to it, with a cold start delay. This is different than pausing the Inference Endpoint with [pause\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint), which would require a manual resume with [resume\_inference\_endpoint()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint).

For convenience, you can also scale an Inference Endpoint to zero using [InferenceEndpoint.scale\_to\_zero()](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.scale_to_zero).

#### set\_space\_sleep\_time

[](#huggingface_hub.HfApi.set_space_sleep_time)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L6936)

( repo\_id: strsleep\_time: inttoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

Parameters

*   [](#huggingface_hub.HfApi.set_space_sleep_time.repo_id)**repo\_id** (`str`) — ID of the repo to update. Example: `"bigcode/in-the-stack"`.
*   [](#huggingface_hub.HfApi.set_space_sleep_time.sleep_time)**sleep\_time** (`int`, _optional_) — Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don’t want your Space to pause (default behavior for upgraded hardware). For free hardware, you can’t configure the sleep time (value is fixed to 48 hours of inactivity). See [https://huggingface.co/docs/hub/spaces-gpus#sleep-time](https://huggingface.co/docs/hub/spaces-gpus#sleep-time) for more details.
*   [](#huggingface_hub.HfApi.set_space_sleep_time.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

export const metadata = 'undefined';

Runtime information about a Space including Space stage and hardware.

Set a custom sleep time for a Space running on upgraded hardware..

Your Space will go to sleep after X seconds of inactivity. You are not billed when your Space is in “sleep” mode. If a new visitor lands on your Space, it will “wake it up”. Only upgraded hardware can have a configurable sleep time. To know more about the sleep stage, please refer to [https://huggingface.co/docs/hub/spaces-gpus#sleep-time](https://huggingface.co/docs/hub/spaces-gpus#sleep-time).

It is also possible to set a custom sleep time when requesting hardware with [request\_space\_hardware()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.request_space_hardware).

#### snapshot\_download

[](#huggingface_hub.HfApi.snapshot_download)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L5271)

( repo\_id: strrepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecache\_dir: Union\[str, Path, None\] = Nonelocal\_dir: Union\[str, Path, None\] = Noneproxies: Optional\[Dict\] = Noneetag\_timeout: float = 10force\_download: bool = Falsetoken: Union\[bool, str, None\] = Nonelocal\_files\_only: bool = Falseallow\_patterns: Optional\[Union\[List\[str\], str\]\] = Noneignore\_patterns: Optional\[Union\[List\[str\], str\]\] = Nonemax\_workers: int = 8tqdm\_class: Optional\[base\_tqdm\] = Nonelocal\_dir\_use\_symlinks: Union\[bool, Literal\['auto'\]\] = 'auto'resume\_download: Optional\[bool\] = None ) → export const metadata = 'undefined';`str`

Expand 14 parameters

Parameters

*   [](#huggingface_hub.HfApi.snapshot_download.repo_id)**repo\_id** (`str`) — A user or an organization name and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.snapshot_download.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if downloading from a dataset or space, `None` or `"model"` if downloading from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.snapshot_download.revision)**revision** (`str`, _optional_) — An optional Git revision id which can be a branch name, a tag, or a commit hash.
*   [](#huggingface_hub.HfApi.snapshot_download.cache_dir)**cache\_dir** (`str`, `Path`, _optional_) — Path to the folder where cached files are stored.
*   [](#huggingface_hub.HfApi.snapshot_download.local_dir)**local\_dir** (`str` or `Path`, _optional_) — If provided, the downloaded files will be placed under this directory.
*   [](#huggingface_hub.HfApi.snapshot_download.proxies)**proxies** (`dict`, _optional_) — Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
*   [](#huggingface_hub.HfApi.snapshot_download.etag_timeout)**etag\_timeout** (`float`, _optional_, defaults to `10`) — When fetching ETag, how many seconds to wait for the server to send data before giving up which is passed to `requests.request`.
*   [](#huggingface_hub.HfApi.snapshot_download.force_download)**force\_download** (`bool`, _optional_, defaults to `False`) — Whether the file should be downloaded even if it already exists in the local cache.
*   [](#huggingface_hub.HfApi.snapshot_download.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.snapshot_download.local_files_only)**local\_files\_only** (`bool`, _optional_, defaults to `False`) — If `True`, avoid downloading the file and return the path to the local cached file if it exists.
*   [](#huggingface_hub.HfApi.snapshot_download.allow_patterns)**allow\_patterns** (`List[str]` or `str`, _optional_) — If provided, only files matching at least one pattern are downloaded.
*   [](#huggingface_hub.HfApi.snapshot_download.ignore_patterns)**ignore\_patterns** (`List[str]` or `str`, _optional_) — If provided, files matching any of the patterns are not downloaded.
*   [](#huggingface_hub.HfApi.snapshot_download.max_workers)**max\_workers** (`int`, _optional_) — Number of concurrent threads to download files (1 thread = 1 file download). Defaults to 8.
*   [](#huggingface_hub.HfApi.snapshot_download.tqdm_class)**tqdm\_class** (`tqdm`, _optional_) — If provided, overwrites the default behavior for the progress bar. Passed argument must inherit from `tqdm.auto.tqdm` or at least mimic its behavior. Note that the `tqdm_class` is not passed to each individual download. Defaults to the custom HF progress bar that can be disabled by setting `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.

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

If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this option, the `cache_dir` will not be used and a `.cache/huggingface/` folder will be created at the root of `local_dir` to store some metadata related to the downloaded files.While this mechanism is not as robust as the main cache-system, it’s optimized for regularly pulling the latest version of a repository.

An alternative would be to clone the repo but this requires git and git-lfs to be installed and properly configured. It is also not possible to filter which files to download when cloning a repository using git.

#### space\_info

[](#huggingface_hub.HfApi.space_info)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2596)

( repo\_id: strrevision: Optional\[str\] = Nonetimeout: Optional\[float\] = Nonefiles\_metadata: bool = Falseexpand: Optional\[List\[ExpandSpaceProperty\_T\]\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[SpaceInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.SpaceInfo)

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.space_info.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.space_info.revision)**revision** (`str`, _optional_) — The revision of the space repository from which to get the information.
*   [](#huggingface_hub.HfApi.space_info.timeout)**timeout** (`float`, _optional_) — Whether to set a timeout for the request to the Hub.
*   [](#huggingface_hub.HfApi.space_info.files_metadata)**files\_metadata** (`bool`, _optional_) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to `False`.
*   [](#huggingface_hub.HfApi.space_info.expand)**expand** (`List[ExpandSpaceProperty_T]`, _optional_) — List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `full` is passed. Possible values are `"author"`, `"cardData"`, `"createdAt"`, `"datasets"`, `"disabled"`, `"lastModified"`, `"likes"`, `"models"`, `"private"`, `"runtime"`, `"sdk"`, `"siblings"`, `"sha"`, `"subdomain"`, `"tags"`, `"trendingScore"`, `"usedStorage"` and `"resourceGroup"`.
*   [](#huggingface_hub.HfApi.space_info.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[SpaceInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.SpaceInfo)

export const metadata = 'undefined';

The space repository information.

Get info on one specific Space on huggingface.co.

Space can be private if you pass an acceptable token.

Raises the following errors:

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) If the revision to download from cannot be found.

#### super\_squash\_history

[](#huggingface_hub.HfApi.super_squash_history)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3296)

( repo\_id: strbranch: Optional\[str\] = Nonecommit\_message: Optional\[str\] = Nonerepo\_type: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = None )

Expand 5 parameters

Parameters

*   [](#huggingface_hub.HfApi.super_squash_history.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.super_squash_history.branch)**branch** (`str`, _optional_) — The branch to squash. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.super_squash_history.commit_message)**commit\_message** (`str`, _optional_) — The commit message to use for the squashed commit.
*   [](#huggingface_hub.HfApi.super_squash_history.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if listing commits from a dataset or a Space, `None` or `"model"` if listing from a model. Default is `None`.
*   [](#huggingface_hub.HfApi.super_squash_history.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) or [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) or [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) — If the branch to squash cannot be found.
*   [BadRequestError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) — If invalid reference for a branch. You cannot squash history on tags.

Squash commit history on a branch for a repo on the Hub.

Squashing the repo history is useful when you know you’ll make hundreds of commits and you don’t want to clutter the history. Squashing commits can only be performed from the head of a branch.

Once squashed, the commit history cannot be retrieved. This is a non-revertible operation.

Once the history of a branch has been squashed, it is not possible to merge it back into another branch since their history will have diverged.

[](#huggingface_hub.HfApi.super_squash_history.example)

Example:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()

\# Create repo
\>>> repo\_id = api.create\_repo("test-squash").repo\_id

\# Make a lot of commits.
\>>> api.upload\_file(repo\_id=repo\_id, path\_in\_repo="file.txt", path\_or\_fileobj=b"content")
\>>> api.upload\_file(repo\_id=repo\_id, path\_in\_repo="lfs.bin", path\_or\_fileobj=b"content")
\>>> api.upload\_file(repo\_id=repo\_id, path\_in\_repo="file.txt", path\_or\_fileobj=b"another\_content")

\# Squash history
\>>> api.super\_squash\_history(repo\_id=repo\_id)

#### unlike

[](#huggingface_hub.HfApi.unlike)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L2279)

( repo\_id: strtoken: Union\[bool, str, None\] = Nonerepo\_type: Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.HfApi.unlike.repo_id)**repo\_id** (`str`) — The repository to unlike. Example: `"user/my-cool-model"`.
*   [](#huggingface_hub.HfApi.unlike.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.unlike.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if unliking a dataset or space, `None` or `"model"` if unliking a model. Default is `None`.

Raises

export const metadata = 'undefined';

[RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

export const metadata = 'undefined';

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If repository is not found (error 404): wrong repo\_id/repo\_type, private but not authenticated or repo does not exist.

Unlike a given repo on the Hub (e.g. remove from favorite list).

To prevent spam usage, it is not possible to `like` a repository from a script.

See also [list\_liked\_repos()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_liked_repos).

[](#huggingface_hub.HfApi.unlike.example)

Example:

Copied

\>>> from huggingface\_hub import list\_liked\_repos, unlike
\>>> "gpt2" in list\_liked\_repos().models \# we assume you have already liked gpt2
True
\>>> unlike("gpt2")
\>>> "gpt2" in list\_liked\_repos().models
False

#### update\_collection\_item

[](#huggingface_hub.HfApi.update_collection_item)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8158)

( collection\_slug: stritem\_object\_id: strnote: Optional\[str\] = Noneposition: Optional\[int\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.update_collection_item.collection_slug)**collection\_slug** (`str`) — Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
*   [](#huggingface_hub.HfApi.update_collection_item.item_object_id)**item\_object\_id** (`str`) — ID of the item in the collection. This is not the id of the item on the Hub (repo\_id or paper id). It must be retrieved from a [CollectionItem](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.CollectionItem) object. Example: `collection.items[0].item_object_id`.
*   [](#huggingface_hub.HfApi.update_collection_item.note)**note** (`str`, _optional_) — A note to attach to the item in the collection. The maximum size for a note is 500 characters.
*   [](#huggingface_hub.HfApi.update_collection_item.position)**position** (`int`, _optional_) — New position of the item in the collection.
*   [](#huggingface_hub.HfApi.update_collection_item.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Update an item in a collection.

[](#huggingface_hub.HfApi.update_collection_item.example)

Example:

Copied

\>>> from huggingface\_hub import get\_collection, update\_collection\_item

\# Get collection first
\>>> collection = get\_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")

\# Update item based on its ID (add note + update position)
\>>> update\_collection\_item(
...     collection\_slug="TheBloke/recent-models-64f9a55bb3115b4f513ec026",
...     item\_object\_id=collection.items\[-1\].item\_object\_id,
...     note="Newly updated model!"
...     position=0,
... )

#### update\_collection\_metadata

[](#huggingface_hub.HfApi.update_collection_metadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7967)

( collection\_slug: strtitle: Optional\[str\] = Nonedescription: Optional\[str\] = Noneposition: Optional\[int\] = Noneprivate: Optional\[bool\] = Nonetheme: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.update_collection_metadata.collection_slug)**collection\_slug** (`str`) — Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
*   [](#huggingface_hub.HfApi.update_collection_metadata.title)**title** (`str`) — Title of the collection to update.
*   [](#huggingface_hub.HfApi.update_collection_metadata.description)**description** (`str`, _optional_) — Description of the collection to update.
*   [](#huggingface_hub.HfApi.update_collection_metadata.position)**position** (`int`, _optional_) — New position of the collection in the list of collections of the user.
*   [](#huggingface_hub.HfApi.update_collection_metadata.private)**private** (`bool`, _optional_) — Whether the collection should be private or not.
*   [](#huggingface_hub.HfApi.update_collection_metadata.theme)**theme** (`str`, _optional_) — Theme of the collection on the Hub.
*   [](#huggingface_hub.HfApi.update_collection_metadata.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Update metadata of a collection on the Hub.

All arguments are optional. Only provided metadata will be updated.

Returns: [Collection](/docs/huggingface_hub/v0.29.2/en/package_reference/collections#huggingface_hub.Collection)

[](#huggingface_hub.HfApi.update_collection_metadata.example)

Example:

Copied

\>>> from huggingface\_hub import update\_collection\_metadata
\>>> collection = update\_collection\_metadata(
...     collection\_slug="username/iccv-2023-64f9a55bb3115b4f513ec026",
...     title="ICCV Oct. 2023"
...     description="Portfolio of models, datasets, papers and demos I presented at ICCV Oct. 2023",
...     private=False,
...     theme="pink",
... )
\>>> collection.slug
"username/iccv-oct-2023-64f9a55bb3115b4f513ec026"
\# ^collection slug got updated but not the trailing ID

#### update\_inference\_endpoint

[](#huggingface_hub.HfApi.update_inference_endpoint)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L7535)

( name: straccelerator: Optional\[str\] = Noneinstance\_size: Optional\[str\] = Noneinstance\_type: Optional\[str\] = Nonemin\_replica: Optional\[int\] = Nonemax\_replica: Optional\[int\] = Nonescale\_to\_zero\_timeout: Optional\[int\] = Nonerepository: Optional\[str\] = Noneframework: Optional\[str\] = Nonerevision: Optional\[str\] = Nonetask: Optional\[str\] = Nonecustom\_image: Optional\[Dict\] = Nonesecrets: Optional\[Dict\[str, str\]\] = Nonenamespace: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

Expand 15 parameters

Parameters

*   [](#huggingface_hub.HfApi.update_inference_endpoint.name)**name** (`str`) — The name of the Inference Endpoint to update.
*   [](#huggingface_hub.HfApi.update_inference_endpoint.accelerator)**accelerator** (`str`, _optional_) — The hardware accelerator to be used for inference (e.g. `"cpu"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.instance_size)**instance\_size** (`str`, _optional_) — The size or type of the instance to be used for hosting the model (e.g. `"x4"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.instance_type)**instance\_type** (`str`, _optional_) — The cloud instance type where the Inference Endpoint will be deployed (e.g. `"intel-icl"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.min_replica)**min\_replica** (`int`, _optional_) — The minimum number of replicas (instances) to keep running for the Inference Endpoint.
*   [](#huggingface_hub.HfApi.update_inference_endpoint.max_replica)**max\_replica** (`int`, _optional_) — The maximum number of replicas (instances) to scale to for the Inference Endpoint.
*   [](#huggingface_hub.HfApi.update_inference_endpoint.scale_to_zero_timeout)**scale\_to\_zero\_timeout** (`int`, _optional_) — The duration in minutes before an inactive endpoint is scaled to zero.
*   [](#huggingface_hub.HfApi.update_inference_endpoint.repository)**repository** (`str`, _optional_) — The name of the model repository associated with the Inference Endpoint (e.g. `"gpt2"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.framework)**framework** (`str`, _optional_) — The machine learning framework used for the model (e.g. `"custom"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.revision)**revision** (`str`, _optional_) — The specific model revision to deploy on the Inference Endpoint (e.g. `"6c0e6080953db56375760c0471a8c5f2929baf11"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.task)**task** (`str`, _optional_) — The task on which to deploy the model (e.g. `"text-classification"`).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.custom_image)**custom\_image** (`Dict`, _optional_) — A custom Docker image to use for the Inference Endpoint. This is useful if you want to deploy an Inference Endpoint running on the `text-generation-inference` (TGI) framework (see examples).
*   [](#huggingface_hub.HfApi.update_inference_endpoint.secrets)**secrets** (`Dict[str, str]`, _optional_) — Secret values to inject in the container environment.
*   [](#huggingface_hub.HfApi.update_inference_endpoint.namespace)**namespace** (`str`, _optional_) — The namespace where the Inference Endpoint will be updated. Defaults to the current user’s namespace.
*   [](#huggingface_hub.HfApi.update_inference_endpoint.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[InferenceEndpoint](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

export const metadata = 'undefined';

information about the updated Inference Endpoint.

Update an Inference Endpoint.

This method allows the update of either the compute configuration, the deployed model, or both. All arguments are optional but at least one must be provided.

For convenience, you can also update an Inference Endpoint using [InferenceEndpoint.update()](/docs/huggingface_hub/v0.29.2/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.update).

#### update\_repo\_settings

[](#huggingface_hub.HfApi.update_repo_settings)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3636)

( repo\_id: strgated: Optional\[Literal\['auto', 'manual', False\]\] = Noneprivate: Optional\[bool\] = Nonetoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = None )

Expand 5 parameters

Parameters

*   [](#huggingface_hub.HfApi.update_repo_settings.repo_id)**repo\_id** (`str`) — A namespace (user or an organization) and a repo name separated by a /.
*   [](#huggingface_hub.HfApi.update_repo_settings.gated)**gated** (`Literal["auto", "manual", False]`, _optional_) — The gated status for the repository. If set to `None` (default), the `gated` setting of the repository won’t be updated.
    
    *   “auto”: The repository is gated, and access requests are automatically approved or denied based on predefined criteria.
    *   “manual”: The repository is gated, and access requests require manual approval.
    *   False : The repository is not gated, and anyone can access it.
    
*   [](#huggingface_hub.HfApi.update_repo_settings.private)**private** (`bool`, _optional_) — Whether the repository should be private.
*   [](#huggingface_hub.HfApi.update_repo_settings.token)**token** (`Union[str, bool, None]`, _optional_) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass False.
*   [](#huggingface_hub.HfApi.update_repo_settings.repo_type)**repo\_type** (`str`, _optional_) — The type of the repository to update settings from (`"model"`, `"dataset"` or `"space"`). Defaults to `"model"`.

Raises

export const metadata = 'undefined';

`ValueError` or [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) or [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

export const metadata = 'undefined';

*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If gated is not one of “auto”, “manual”, or False.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If repo\_type is not one of the values in constants.REPO\_TYPES.
*   [HfHubHTTPError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) — If the request to the Hugging Face Hub API fails.
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) — If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

Update the settings of a repository, including gated access and visibility.

To give more control over how repos are used, the Hub allows repo authors to enable access requests for their repos, and also to set the visibility of the repo to private.

#### update\_repo\_visibility

[](#huggingface_hub.HfApi.update_repo_visibility)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L3581)

( repo\_id: strprivate: bool = Falsetoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.HfApi.update_repo_visibility.repo_id)**repo\_id** (`str`, _optional_) — A namespace (user or an organization) and a repo name separated by a `/`.
*   [](#huggingface_hub.HfApi.update_repo_visibility.private)**private** (`bool`, _optional_, defaults to `False`) — Whether the repository should be private.
*   [](#huggingface_hub.HfApi.update_repo_visibility.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.update_repo_visibility.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.

Update the visibility setting of a repository.

Deprecated. Use `update_repo_settings` instead.

Raises the following errors:

*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.

#### update\_webhook

[](#huggingface_hub.HfApi.update_webhook)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L8876)

( webhook\_id: strurl: Optional\[str\] = Nonewatched: Optional\[List\[Union\[Dict, WebhookWatchedItem\]\]\] = Nonedomains: Optional\[List\[constants.WEBHOOK\_DOMAIN\_T\]\] = Nonesecret: Optional\[str\] = Nonetoken: Union\[bool, str, None\] = None ) → export const metadata = 'undefined';[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

Expand 6 parameters

Parameters

*   [](#huggingface_hub.HfApi.update_webhook.webhook_id)**webhook\_id** (`str`) — The unique identifier of the webhook to be updated.
*   [](#huggingface_hub.HfApi.update_webhook.url)**url** (`str`, optional) — The URL to which the payload will be sent.
*   [](#huggingface_hub.HfApi.update_webhook.watched)**watched** (`List[WebhookWatchedItem]`, optional) — List of items to watch. It can be users, orgs, models, datasets, or spaces. Refer to [WebhookWatchedItem](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookWatchedItem) for more details. Watched items can also be provided as plain dictionaries.
*   [](#huggingface_hub.HfApi.update_webhook.domains)**domains** (`List[Literal["repo", "discussion"]]`, optional) — The domains to watch. This can include “repo”, “discussion”, or both.
*   [](#huggingface_hub.HfApi.update_webhook.secret)**secret** (`str`, optional) — A secret to sign the payload with, providing an additional layer of security.
*   [](#huggingface_hub.HfApi.update_webhook.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Returns

export const metadata = 'undefined';

[WebhookInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

export const metadata = 'undefined';

Info about the updated webhook.

Update an existing webhook.

[](#huggingface_hub.HfApi.update_webhook.example)

Example:

Copied

\>>> from huggingface\_hub import update\_webhook
\>>> updated\_payload = update\_webhook(
...     webhook\_id="654bbbc16f2ec14d77f109cc",
...     url="https://new.webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
...     watched=\[{"type": "user", "name": "julien-c"}, {"type": "org", "name": "HuggingFaceH4"}\],
...     domains=\["repo"\],
...     secret="my-secret",
... )
\>>> print(updated\_payload)
WebhookInfo(
    id\="654bbbc16f2ec14d77f109cc",
    url="https://new.webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    watched=\[WebhookWatchedItem(type\="user", name="julien-c"), WebhookWatchedItem(type\="org", name="HuggingFaceH4")\],
    domains=\["repo"\],
    secret="my-secret",
    disabled=False,

#### upload\_file

[](#huggingface_hub.HfApi.upload_file)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4296)

( path\_or\_fileobj: Union\[str, Path, bytes, BinaryIO\]path\_in\_repo: strrepo\_id: strtoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecommit\_message: Optional\[str\] = Nonecommit\_description: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Noneparent\_commit: Optional\[str\] = Nonerun\_as\_future: bool = False ) → export const metadata = 'undefined';[CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) or `Future`

Expand 11 parameters

Parameters

*   [](#huggingface_hub.HfApi.upload_file.path_or_fileobj)**path\_or\_fileobj** (`str`, `Path`, `bytes`, or `IO`) — Path to a file on the local machine or binary data stream / fileobj / buffer.
*   [](#huggingface_hub.HfApi.upload_file.path_in_repo)**path\_in\_repo** (`str`) — Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
*   [](#huggingface_hub.HfApi.upload_file.repo_id)**repo\_id** (`str`) — The repository to which the file will be uploaded, for example: `"username/custom_transformers"`
*   [](#huggingface_hub.HfApi.upload_file.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.upload_file.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.upload_file.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.upload_file.commit_message)**commit\_message** (`str`, _optional_) — The summary / title / first line of the generated commit
*   [](#huggingface_hub.HfApi.upload_file.commit_description)**commit\_description** (`str` _optional_) — The description of the generated commit
*   [](#huggingface_hub.HfApi.upload_file.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened against this branch. If `revision` is set and is not a branch name (example: a commit oid), an `RevisionNotFoundError` is returned by the server.
*   [](#huggingface_hub.HfApi.upload_file.parent_commit)**parent\_commit** (`str`, _optional_) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.
*   [](#huggingface_hub.HfApi.upload_file.run_as_future)**run\_as\_future** (`bool`, _optional_) — Whether or not to run this method in the background. Background jobs are run sequentially without blocking the main thread. Passing `run_as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) object. Defaults to `False`.

Returns

export const metadata = 'undefined';

[CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) or `Future`

export const metadata = 'undefined';

Instance of [CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) containing information about the newly created commit (commit hash, commit url, pr url, commit message,…). If `run_as_future=True` is passed, returns a Future object which will contain the result when executed.

Upload a local file (up to 50 GB) to the given repo. The upload is done through a HTTP post request, and doesn’t require git or git-lfs to be installed.

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid
*   [RepositoryNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to `private` and you do not have access.
*   [RevisionNotFoundError](/docs/huggingface_hub/v0.29.2/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) If the revision to download from cannot be found.

`upload_file` assumes that the repo already exists on the Hub. If you get a Client error 404, please make sure you are authenticated and that `repo_id` and `repo_type` are set correctly. If repo does not exist, create it first using [create\_repo()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo).

[](#huggingface_hub.HfApi.upload_file.example)

Example:

Copied

\>>> from huggingface\_hub import upload\_file

\>>> with open("./local/filepath", "rb") as fobj:
...     upload\_file(
...         path\_or\_fileobj=fileobj,
...         path\_in\_repo="remote/file/path.h5",
...         repo\_id="username/my-dataset",
...         repo\_type="dataset",
...         token="my\_token",
...     )
"https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

\>>> upload\_file(
...     path\_or\_fileobj=".\\\\local\\\\file\\\\path",
...     path\_in\_repo="remote/file/path.h5",
...     repo\_id="username/my-model",
...     token="my\_token",
... )
"https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"

\>>> upload\_file(
...     path\_or\_fileobj=".\\\\local\\\\file\\\\path",
...     path\_in\_repo="remote/file/path.h5",
...     repo\_id="username/my-model",
...     token="my\_token",
...     create\_pr=True,
... )
"https://huggingface.co/username/my-model/blob/refs%2Fpr%2F1/remote/file/path.h5"

#### upload\_folder

[](#huggingface_hub.HfApi.upload_folder)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4504)

( repo\_id: strfolder\_path: Union\[str, Path\]path\_in\_repo: Optional\[str\] = Nonecommit\_message: Optional\[str\] = Nonecommit\_description: Optional\[str\] = Nonetoken: Union\[str, bool, None\] = Nonerepo\_type: Optional\[str\] = Nonerevision: Optional\[str\] = Nonecreate\_pr: Optional\[bool\] = Noneparent\_commit: Optional\[str\] = Noneallow\_patterns: Optional\[Union\[List\[str\], str\]\] = Noneignore\_patterns: Optional\[Union\[List\[str\], str\]\] = Nonedelete\_patterns: Optional\[Union\[List\[str\], str\]\] = Nonerun\_as\_future: bool = False ) → export const metadata = 'undefined';[CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) or `Future`

Expand 14 parameters

Parameters

*   [](#huggingface_hub.HfApi.upload_folder.repo_id)**repo\_id** (`str`) — The repository to which the file will be uploaded, for example: `"username/custom_transformers"`
*   [](#huggingface_hub.HfApi.upload_folder.folder_path)**folder\_path** (`str` or `Path`) — Path to the folder to upload on the local file system
*   [](#huggingface_hub.HfApi.upload_folder.path_in_repo)**path\_in\_repo** (`str`, _optional_) — Relative path of the directory in the repo, for example: `"checkpoints/1fec34a/results"`. Will default to the root folder of the repository.
*   [](#huggingface_hub.HfApi.upload_folder.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.
*   [](#huggingface_hub.HfApi.upload_folder.repo_type)**repo\_type** (`str`, _optional_) — Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to a model. Default is `None`.
*   [](#huggingface_hub.HfApi.upload_folder.revision)**revision** (`str`, _optional_) — The git revision to commit from. Defaults to the head of the `"main"` branch.
*   [](#huggingface_hub.HfApi.upload_folder.commit_message)**commit\_message** (`str`, _optional_) — The summary / title / first line of the generated commit. Defaults to: `f"Upload {path_in_repo} with huggingface_hub"`
*   [](#huggingface_hub.HfApi.upload_folder.commit_description)**commit\_description** (`str` _optional_) — The description of the generated commit
*   [](#huggingface_hub.HfApi.upload_folder.create_pr)**create\_pr** (`boolean`, _optional_) — Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened against this branch. If `revision` is set and is not a branch name (example: a commit oid), an `RevisionNotFoundError` is returned by the server.
*   [](#huggingface_hub.HfApi.upload_folder.parent_commit)**parent\_commit** (`str`, _optional_) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.
*   [](#huggingface_hub.HfApi.upload_folder.allow_patterns)**allow\_patterns** (`List[str]` or `str`, _optional_) — If provided, only files matching at least one pattern are uploaded.
*   [](#huggingface_hub.HfApi.upload_folder.ignore_patterns)**ignore\_patterns** (`List[str]` or `str`, _optional_) — If provided, files matching any of the patterns are not uploaded.
*   [](#huggingface_hub.HfApi.upload_folder.delete_patterns)**delete\_patterns** (`List[str]` or `str`, _optional_) — If provided, remote files matching any of the patterns will be deleted from the repo while committing new files. This is useful if you don’t know which files have already been uploaded. Note: to avoid discrepancies the `.gitattributes` file is not deleted even if it matches the pattern.
*   [](#huggingface_hub.HfApi.upload_folder.run_as_future)**run\_as\_future** (`bool`, _optional_) — Whether or not to run this method in the background. Background jobs are run sequentially without blocking the main thread. Passing `run_as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) object. Defaults to `False`.

Returns

export const metadata = 'undefined';

[CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) or `Future`

export const metadata = 'undefined';

Instance of [CommitInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitInfo) containing information about the newly created commit (commit hash, commit url, pr url, commit message,…). If `run_as_future=True` is passed, returns a Future object which will contain the result when executed.

Upload a local folder to the given repo. The upload is done through a HTTP requests, and doesn’t require git or git-lfs to be installed.

The structure of the folder will be preserved. Files with the same name already present in the repository will be overwritten. Others will be left untouched.

Use the `allow_patterns` and `ignore_patterns` arguments to specify which files to upload. These parameters accept either a single pattern or a list of patterns. Patterns are Standard Wildcards (globbing patterns) as documented [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). If both `allow_patterns` and `ignore_patterns` are provided, both constraints apply. By default, all files from the folder are uploaded.

Use the `delete_patterns` argument to specify remote files you want to delete. Input type is the same as for `allow_patterns` (see above). If `path_in_repo` is also provided, the patterns are matched against paths relative to this folder. For example, `upload_folder(..., path_in_repo="experiment", delete_patterns="logs/*")` will delete any remote file under `./experiment/logs/`. Note that the `.gitattributes` file will not be deleted even if it matches the patterns.

Any `.git/` folder present in any subdirectory will be ignored. However, please be aware that the `.gitignore` file is not taken into account.

Uses `HfApi.create_commit` under the hood.

Raises the following errors:

*   [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) if the HuggingFace API returned an error
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid

`upload_folder` assumes that the repo already exists on the Hub. If you get a Client error 404, please make sure you are authenticated and that `repo_id` and `repo_type` are set correctly. If repo does not exist, create it first using [create\_repo()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo).

When dealing with a large folder (thousands of files or hundreds of GB), we recommend using [upload\_large\_folder()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_large_folder) instead.

[](#huggingface_hub.HfApi.upload_folder.example)

Example:

Copied

\# Upload checkpoints folder except the log files
\>>> upload\_folder(
...     folder\_path="local/checkpoints",
...     path\_in\_repo="remote/experiment/checkpoints",
...     repo\_id="username/my-dataset",
...     repo\_type="datasets",
...     token="my\_token",
...     ignore\_patterns="\*\*/logs/\*.txt",
... )
\# "https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

\# Upload checkpoints folder including logs while deleting existing logs from the repo
\# Useful if you don't know exactly which log files have already being pushed
\>>> upload\_folder(
...     folder\_path="local/checkpoints",
...     path\_in\_repo="remote/experiment/checkpoints",
...     repo\_id="username/my-dataset",
...     repo\_type="datasets",
...     token="my\_token",
...     delete\_patterns="\*\*/logs/\*.txt",
... )
"https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

\# Upload checkpoints folder while creating a PR
\>>> upload\_folder(
...     folder\_path="local/checkpoints",
...     path\_in\_repo="remote/experiment/checkpoints",
...     repo\_id="username/my-dataset",
...     repo\_type="datasets",
...     token="my\_token",
...     create\_pr=True,
... )
"https://huggingface.co/datasets/username/my-dataset/tree/refs%2Fpr%2F1/remote/experiment/checkpoints"

#### upload\_large\_folder

[](#huggingface_hub.HfApi.upload_large_folder)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L4971)

( repo\_id: strfolder\_path: Union\[str, Path\]repo\_type: strrevision: Optional\[str\] = Noneprivate: Optional\[bool\] = Noneallow\_patterns: Optional\[Union\[List\[str\], str\]\] = Noneignore\_patterns: Optional\[Union\[List\[str\], str\]\] = Nonenum\_workers: Optional\[int\] = Noneprint\_report: bool = Trueprint\_report\_every: int = 60 )

Expand 10 parameters

Parameters

*   [](#huggingface_hub.HfApi.upload_large_folder.repo_id)**repo\_id** (`str`) — The repository to which the file will be uploaded. E.g. `"HuggingFaceTB/smollm-corpus"`.
*   [](#huggingface_hub.HfApi.upload_large_folder.folder_path)**folder\_path** (`str` or `Path`) — Path to the folder to upload on the local file system.
*   [](#huggingface_hub.HfApi.upload_large_folder.repo_type)**repo\_type** (`str`) — Type of the repository. Must be one of `"model"`, `"dataset"` or `"space"`. Unlike in all other `HfApi` methods, `repo_type` is explicitly required here. This is to avoid any mistake when uploading a large folder to the Hub, and therefore prevent from having to re-upload everything.
*   [](#huggingface_hub.HfApi.upload_large_folder.revision)**revision** (`str`, `optional`) — The branch to commit to. If not provided, the `main` branch will be used.
*   [](#huggingface_hub.HfApi.upload_large_folder.private)**private** (`bool`, `optional`) — Whether the repository should be private. If `None` (default), the repo will be public unless the organization’s default is private.
*   [](#huggingface_hub.HfApi.upload_large_folder.allow_patterns)**allow\_patterns** (`List[str]` or `str`, _optional_) — If provided, only files matching at least one pattern are uploaded.
*   [](#huggingface_hub.HfApi.upload_large_folder.ignore_patterns)**ignore\_patterns** (`List[str]` or `str`, _optional_) — If provided, files matching any of the patterns are not uploaded.
*   [](#huggingface_hub.HfApi.upload_large_folder.num_workers)**num\_workers** (`int`, _optional_) — Number of workers to start. Defaults to `os.cpu_count() - 2` (minimum 2). A higher number of workers may speed up the process if your machine allows it. However, on machines with a slower connection, it is recommended to keep the number of workers low to ensure better resumability. Indeed, partially uploaded files will have to be completely re-uploaded if the process is interrupted.
*   [](#huggingface_hub.HfApi.upload_large_folder.print_report)**print\_report** (`bool`, _optional_) — Whether to print a report of the upload progress. Defaults to True. Report is printed to `sys.stdout` every X seconds (60 by defaults) and overwrites the previous report.
*   [](#huggingface_hub.HfApi.upload_large_folder.print_report_every)**print\_report\_every** (`int`, _optional_) — Frequency at which the report is printed. Defaults to 60 seconds.

Upload a large folder to the Hub in the most resilient way possible.

Several workers are started to upload files in an optimized way. Before being committed to a repo, files must be hashed and be pre-uploaded if they are LFS files. Workers will perform these tasks for each file in the folder. At each step, some metadata information about the upload process is saved in the folder under `.cache/.huggingface/` to be able to resume the process if interrupted. The whole process might result in several commits.

A few things to keep in mind:

*   Repository limits still apply: [https://huggingface.co/docs/hub/repositories-recommendations](https://huggingface.co/docs/hub/repositories-recommendations)
*   Do not start several processes in parallel.
*   You can interrupt and resume the process at any time.
*   Do not upload the same folder to several repositories. If you need to do so, you must delete the local `.cache/.huggingface/` folder first.

While being much more robust to upload large folders, `upload_large_folder` is more limited than [upload\_folder()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) feature-wise. In practice:

*   you cannot set a custom `path_in_repo`. If you want to upload to a subfolder, you need to set the proper structure locally.
*   you cannot set a custom `commit_message` and `commit_description` since multiple commits are created.
*   you cannot delete from the repo while uploading. Please make a separate commit first.
*   you cannot create a PR directly. Please create a PR first (from the UI or using [create\_pull\_request()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_pull_request)) and then commit to it by passing `revision`.

**Technical details:**

`upload_large_folder` process is as follow:

1.  (Check parameters and setup.)
2.  Create repo if missing.
3.  List local files to upload.
4.  Start workers. Workers can perform the following tasks:
    *   Hash a file.
    *   Get upload mode (regular or LFS) for a list of files.
    *   Pre-upload an LFS file.
    *   Commit a bunch of files. Once a worker finishes a task, it will move on to the next task based on the priority list (see below) until all files are uploaded and committed.
5.  While workers are up, regularly print a report to sys.stdout.

Order of priority:

1.  Commit if more than 5 minutes since last commit attempt (and at least 1 file).
2.  Commit if at least 150 files are ready to commit.
3.  Get upload mode if at least 10 files have been hashed.
4.  Pre-upload LFS file if at least 1 file and no worker is pre-uploading.
5.  Hash file if at least 1 file and no worker is hashing.
6.  Get upload mode if at least 1 file and no worker is getting upload mode.
7.  Pre-upload LFS file if at least 1 file (exception: if hf\_transfer is enabled, only 1 worker can preupload LFS at a time).
8.  Hash file if at least 1 file to hash.
9.  Get upload mode if at least 1 file to get upload mode.
10.  Commit if at least 1 file to commit and at least 1 min since last commit attempt.
11.  Commit if at least 1 file to commit and all other queues are empty.

Special rules:

*   If `hf_transfer` is enabled, only 1 LFS uploader at a time. Otherwise the CPU would be bloated by `hf_transfer`.
*   Only one worker can commit at a time.
*   If no tasks are available, the worker waits for 10 seconds before checking again.

#### whoami

[](#huggingface_hub.HfApi.whoami)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1645)

( token: Union\[bool, str, None\] = None )

Parameters

*   [](#huggingface_hub.HfApi.whoami.token)**token** (Union\[bool, str, None\], optional) — A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see [https://huggingface.co/docs/huggingface\_hub/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). To disable authentication, pass `False`.

Call HF API to know “whoami”.

[](#api-dataclasses)API Dataclasses
-----------------------------------

### [](#huggingface_hub.hf_api.AccessRequest)AccessRequest

### class huggingface\_hub.hf\_api.AccessRequest

[](#huggingface_hub.hf_api.AccessRequest)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L444)

( username: strfullname: stremail: Optional\[str\]timestamp: datetimestatus: Literal\['pending', 'accepted', 'rejected'\]fields: Optional\[Dict\[str, Any\]\] = None )

Parameters

*   [](#huggingface_hub.hf_api.AccessRequest.username)**username** (`str`) — Username of the user who requested access.
*   [](#huggingface_hub.hf_api.AccessRequest.fullname)**fullname** (`str`) — Fullname of the user who requested access.
*   [](#huggingface_hub.hf_api.AccessRequest.email)**email** (`Optional[str]`) — Email of the user who requested access. Can only be `None` in the /accepted list if the user was granted access manually.
*   [](#huggingface_hub.hf_api.AccessRequest.timestamp)**timestamp** (`datetime`) — Timestamp of the request.
*   [](#huggingface_hub.hf_api.AccessRequest.status)**status** (`Literal["pending", "accepted", "rejected"]`) — Status of the request. Can be one of `["pending", "accepted", "rejected"]`.
*   [](#huggingface_hub.hf_api.AccessRequest.fields)**fields** (`Dict[str, Any]`, _optional_) — Additional fields filled by the user in the gate form.

Data structure containing information about a user access request.

### [](#huggingface_hub.CommitInfo)CommitInfo

### class huggingface\_hub.CommitInfo

[](#huggingface_hub.CommitInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L364)

( \*argscommit\_url: str\_url: Optional\[str\] = None\*\*kwargs )

Parameters

*   [](#huggingface_hub.CommitInfo.commit_url)**commit\_url** (`str`) — Url where to find the commit.
*   [](#huggingface_hub.CommitInfo.commit_message)**commit\_message** (`str`) — The summary (first line) of the commit that has been created.
*   [](#huggingface_hub.CommitInfo.commit_description)**commit\_description** (`str`) — Description of the commit that has been created. Can be empty.
*   [](#huggingface_hub.CommitInfo.oid)**oid** (`str`) — Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.
*   [](#huggingface_hub.CommitInfo.pr_url)**pr\_url** (`str`, _optional_) — Url to the PR that has been created, if any. Populated when `create_pr=True` is passed.
*   [](#huggingface_hub.CommitInfo.pr_revision)**pr\_revision** (`str`, _optional_) — Revision of the PR that has been created, if any. Populated when `create_pr=True` is passed. Example: `"refs/pr/1"`.
*   [](#huggingface_hub.CommitInfo.pr_num)**pr\_num** (`int`, _optional_) — Number of the PR discussion that has been created, if any. Populated when `create_pr=True` is passed. Can be passed as `discussion_num` in [get\_discussion\_details()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.get_discussion_details). Example: `1`.
*   [](#huggingface_hub.CommitInfo.repo_url)**repo\_url** (`RepoUrl`) — Repo URL of the commit containing info like repo\_id, repo\_type, etc.
*   [](#huggingface_hub.CommitInfo._url)**\_url** (`str`, _optional_) — Legacy url for `str` compatibility. Can be the url to the uploaded file on the Hub (if returned by [upload\_file()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)), to the uploaded folder on the Hub (if returned by [upload\_folder()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)) or to the commit on the Hub (if returned by [create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)). Defaults to `commit_url`. It is deprecated to use this attribute. Please use `commit_url` instead.

Data structure containing information about a newly created commit.

Returned by any method that creates a commit on the Hub: [create\_commit()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit), [upload\_file()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file), [upload\_folder()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder), [delete\_file()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.delete_file), [delete\_folder()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.delete_folder). It inherits from `str` for backward compatibility but using methods specific to `str` is deprecated.

### [](#huggingface_hub.DatasetInfo)DatasetInfo

### class huggingface\_hub.DatasetInfo

[](#huggingface_hub.DatasetInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L900)

( \*\*kwargs )

Expand 16 parameters

Parameters

*   [](#huggingface_hub.DatasetInfo.id)**id** (`str`) — ID of dataset.
*   [](#huggingface_hub.DatasetInfo.author)**author** (`str`) — Author of the dataset.
*   [](#huggingface_hub.DatasetInfo.sha)**sha** (`str`) — Repo SHA at this particular revision.
*   [](#huggingface_hub.DatasetInfo.created_at)**created\_at** (`datetime`, _optional_) — Date of creation of the repo on the Hub. Note that the lowest value is `2022-03-02T23:29:04.000Z`, corresponding to the date when we began to store creation dates.
*   [](#huggingface_hub.DatasetInfo.last_modified)**last\_modified** (`datetime`, _optional_) — Date of last commit to the repo.
*   [](#huggingface_hub.DatasetInfo.private)**private** (`bool`) — Is the repo private.
*   [](#huggingface_hub.DatasetInfo.disabled)**disabled** (`bool`, _optional_) — Is the repo disabled.
*   [](#huggingface_hub.DatasetInfo.gated)**gated** (`Literal["auto", "manual", False]`, _optional_) — Is the repo gated. If so, whether there is manual or automatic approval.
*   [](#huggingface_hub.DatasetInfo.downloads)**downloads** (`int`) — Number of downloads of the dataset over the last 30 days.
*   [](#huggingface_hub.DatasetInfo.downloads_all_time)**downloads\_all\_time** (`int`) — Cumulated number of downloads of the model since its creation.
*   [](#huggingface_hub.DatasetInfo.likes)**likes** (`int`) — Number of likes of the dataset.
*   [](#huggingface_hub.DatasetInfo.tags)**tags** (`List[str]`) — List of tags of the dataset.
*   [](#huggingface_hub.DatasetInfo.card_data)**card\_data** (`DatasetCardData`, _optional_) — Model Card Metadata as a [huggingface\_hub.repocard\_data.DatasetCardData](/docs/huggingface_hub/v0.29.2/en/package_reference/cards#huggingface_hub.DatasetCardData) object.
*   [](#huggingface_hub.DatasetInfo.siblings)**siblings** (`List[RepoSibling]`) — List of [huggingface\_hub.hf\_api.RepoSibling](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.hf_api.RepoSibling) objects that constitute the dataset.
*   [](#huggingface_hub.DatasetInfo.paperswithcode_id)**paperswithcode\_id** (`str`, _optional_) — Papers with code ID of the dataset.
*   [](#huggingface_hub.DatasetInfo.trending_score)**trending\_score** (`int`, _optional_) — Trending score of the dataset.

Contains information about a dataset on the Hub.

Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made. In general, the more specific the query, the more information is returned. On the contrary, when listing datasets using [list\_datasets()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) only a subset of the attributes are returned.

### [](#huggingface_hub.GitRefInfo)GitRefInfo

### class huggingface\_hub.GitRefInfo

[](#huggingface_hub.GitRefInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1246)

( name: strref: strtarget\_commit: str )

Parameters

*   [](#huggingface_hub.GitRefInfo.name)**name** (`str`) — Name of the reference (e.g. tag name or branch name).
*   [](#huggingface_hub.GitRefInfo.ref)**ref** (`str`) — Full git ref on the Hub (e.g. `"refs/heads/main"` or `"refs/tags/v1.0"`).
*   [](#huggingface_hub.GitRefInfo.target_commit)**target\_commit** (`str`) — OID of the target commit for the ref (e.g. `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`)

Contains information about a git reference for a repo on the Hub.

### [](#huggingface_hub.GitCommitInfo)GitCommitInfo

### class huggingface\_hub.GitCommitInfo

[](#huggingface_hub.GitCommitInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1291)

( commit\_id: strauthors: List\[str\]created\_at: datetimetitle: strmessage: strformatted\_title: Optional\[str\]formatted\_message: Optional\[str\] )

Parameters

*   [](#huggingface_hub.GitCommitInfo.commit_id)**commit\_id** (`str`) — OID of the commit (e.g. `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`)
*   [](#huggingface_hub.GitCommitInfo.authors)**authors** (`List[str]`) — List of authors of the commit.
*   [](#huggingface_hub.GitCommitInfo.created_at)**created\_at** (`datetime`) — Datetime when the commit was created.
*   [](#huggingface_hub.GitCommitInfo.title)**title** (`str`) — Title of the commit. This is a free-text value entered by the authors.
*   [](#huggingface_hub.GitCommitInfo.message)**message** (`str`) — Description of the commit. This is a free-text value entered by the authors.
*   [](#huggingface_hub.GitCommitInfo.formatted_title)**formatted\_title** (`str`) — Title of the commit formatted as HTML. Only returned if `formatted=True` is set.
*   [](#huggingface_hub.GitCommitInfo.formatted_message)**formatted\_message** (`str`) — Description of the commit formatted as HTML. Only returned if `formatted=True` is set.

Contains information about a git commit for a repo on the Hub. Check out [list\_repo\_commits()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_commits) for more details.

### [](#huggingface_hub.GitRefs)GitRefs

### class huggingface\_hub.GitRefs

[](#huggingface_hub.GitRefs)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1265)

( branches: List\[GitRefInfo\]converts: List\[GitRefInfo\]tags: List\[GitRefInfo\]pull\_requests: Optional\[List\[GitRefInfo\]\] = None )

Parameters

*   [](#huggingface_hub.GitRefs.branches)**branches** (`List[GitRefInfo]`) — A list of [GitRefInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitRefInfo) containing information about branches on the repo.
*   [](#huggingface_hub.GitRefs.converts)**converts** (`List[GitRefInfo]`) — A list of [GitRefInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitRefInfo) containing information about “convert” refs on the repo. Converts are refs used (internally) to push preprocessed data in Dataset repos.
*   [](#huggingface_hub.GitRefs.tags)**tags** (`List[GitRefInfo]`) — A list of [GitRefInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitRefInfo) containing information about tags on the repo.
*   [](#huggingface_hub.GitRefs.pull_requests)**pull\_requests** (`List[GitRefInfo]`, _optional_) — A list of [GitRefInfo](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.GitRefInfo) containing information about pull requests on the repo. Only returned if `include_prs=True` is set.

Contains information about all git references for a repo on the Hub.

Object is returned by [list\_repo\_refs()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_refs).

### [](#huggingface_hub.ModelInfo)ModelInfo

### class huggingface\_hub.ModelInfo

[](#huggingface_hub.ModelInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L714)

( \*\*kwargs )

Expand 28 parameters

Parameters

*   [](#huggingface_hub.ModelInfo.id)**id** (`str`) — ID of model.
*   [](#huggingface_hub.ModelInfo.author)**author** (`str`, _optional_) — Author of the model.
*   [](#huggingface_hub.ModelInfo.sha)**sha** (`str`, _optional_) — Repo SHA at this particular revision.
*   [](#huggingface_hub.ModelInfo.created_at)**created\_at** (`datetime`, _optional_) — Date of creation of the repo on the Hub. Note that the lowest value is `2022-03-02T23:29:04.000Z`, corresponding to the date when we began to store creation dates.
*   [](#huggingface_hub.ModelInfo.last_modified)**last\_modified** (`datetime`, _optional_) — Date of last commit to the repo.
*   [](#huggingface_hub.ModelInfo.private)**private** (`bool`) — Is the repo private.
*   [](#huggingface_hub.ModelInfo.disabled)**disabled** (`bool`, _optional_) — Is the repo disabled.
*   [](#huggingface_hub.ModelInfo.downloads)**downloads** (`int`) — Number of downloads of the model over the last 30 days.
*   [](#huggingface_hub.ModelInfo.downloads_all_time)**downloads\_all\_time** (`int`) — Cumulated number of downloads of the model since its creation.
*   [](#huggingface_hub.ModelInfo.gated)**gated** (`Literal["auto", "manual", False]`, _optional_) — Is the repo gated. If so, whether there is manual or automatic approval.
*   [](#huggingface_hub.ModelInfo.gguf)**gguf** (`Dict`, _optional_) — GGUF information of the model.
*   [](#huggingface_hub.ModelInfo.inference)**inference** (`Literal["cold", "frozen", "warm"]`, _optional_) — Status of the model on the inference API. Warm models are available for immediate use. Cold models will be loaded on first inference call. Frozen models are not available in Inference API.
*   [](#huggingface_hub.ModelInfo.inference_provider_mapping)**inference\_provider\_mapping** (`Dict`, _optional_) — Model’s inference provider mapping.
*   [](#huggingface_hub.ModelInfo.likes)**likes** (`int`) — Number of likes of the model.
*   [](#huggingface_hub.ModelInfo.library_name)**library\_name** (`str`, _optional_) — Library associated with the model.
*   [](#huggingface_hub.ModelInfo.tags)**tags** (`List[str]`) — List of tags of the model. Compared to `card_data.tags`, contains extra tags computed by the Hub (e.g. supported libraries, model’s arXiv).
*   [](#huggingface_hub.ModelInfo.pipeline_tag)**pipeline\_tag** (`str`, _optional_) — Pipeline tag associated with the model.
*   [](#huggingface_hub.ModelInfo.mask_token)**mask\_token** (`str`, _optional_) — Mask token used by the model.
*   [](#huggingface_hub.ModelInfo.widget_data)**widget\_data** (`Any`, _optional_) — Widget data associated with the model.
*   [](#huggingface_hub.ModelInfo.model_index)**model\_index** (`Dict`, _optional_) — Model index for evaluation.
*   [](#huggingface_hub.ModelInfo.config)**config** (`Dict`, _optional_) — Model configuration.
*   [](#huggingface_hub.ModelInfo.transformers_info)**transformers\_info** (`TransformersInfo`, _optional_) — Transformers-specific info (auto class, processor, etc.) associated with the model.
*   [](#huggingface_hub.ModelInfo.trending_score)**trending\_score** (`int`, _optional_) — Trending score of the model.
*   [](#huggingface_hub.ModelInfo.card_data)**card\_data** (`ModelCardData`, _optional_) — Model Card Metadata as a [huggingface\_hub.repocard\_data.ModelCardData](/docs/huggingface_hub/v0.29.2/en/package_reference/cards#huggingface_hub.ModelCardData) object.
*   [](#huggingface_hub.ModelInfo.siblings)**siblings** (`List[RepoSibling]`) — List of [huggingface\_hub.hf\_api.RepoSibling](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.hf_api.RepoSibling) objects that constitute the model.
*   [](#huggingface_hub.ModelInfo.spaces)**spaces** (`List[str]`, _optional_) — List of spaces using the model.
*   [](#huggingface_hub.ModelInfo.safetensors)**safetensors** (`SafeTensorsInfo`, _optional_) — Model’s safetensors information.
*   [](#huggingface_hub.ModelInfo.security_repo_status)**security\_repo\_status** (`Dict`, _optional_) — Model’s security scan status.

Contains information about a model on the Hub.

Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made. In general, the more specific the query, the more information is returned. On the contrary, when listing models using [list\_models()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_models) only a subset of the attributes are returned.

### [](#huggingface_hub.hf_api.RepoSibling)RepoSibling

### class huggingface\_hub.hf\_api.RepoSibling

[](#huggingface_hub.hf_api.RepoSibling)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L577)

( rfilename: strsize: Optional\[int\] = Noneblob\_id: Optional\[str\] = Nonelfs: Optional\[BlobLfsInfo\] = None )

Parameters

*   [](#huggingface_hub.hf_api.RepoSibling.rfilename)**rfilename** (str) — file name, relative to the repo root.
*   [](#huggingface_hub.hf_api.RepoSibling.size)**size** (`int`, _optional_) — The file’s size, in bytes. This attribute is defined when `files_metadata` argument of [repo\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.repo_info) is set to `True`. It’s `None` otherwise.
*   [](#huggingface_hub.hf_api.RepoSibling.blob_id)**blob\_id** (`str`, _optional_) — The file’s git OID. This attribute is defined when `files_metadata` argument of [repo\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.repo_info) is set to `True`. It’s `None` otherwise.
*   [](#huggingface_hub.hf_api.RepoSibling.lfs)**lfs** (`BlobLfsInfo`, _optional_) — The file’s LFS metadata. This attribute is defined when`files_metadata` argument of [repo\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.repo_info) is set to `True` and the file is stored with Git LFS. It’s `None` otherwise.

Contains basic information about a repo file inside a repo on the Hub.

All attributes of this class are optional except `rfilename`. This is because only the file names are returned when listing repositories on the Hub (with [list\_models()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_models), [list\_datasets()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) or [list\_spaces()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces)). If you need more information like file size, blob id or lfs details, you must request them specifically from one repo at a time (using [model\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.model_info), [dataset\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info) or [space\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.space_info)) as it adds more constraints on the backend server to retrieve these.

### [](#huggingface_hub.hf_api.RepoFile)RepoFile

### class huggingface\_hub.hf\_api.RepoFile

[](#huggingface_hub.hf_api.RepoFile)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L612)

( \*\*kwargs )

Parameters

*   [](#huggingface_hub.hf_api.RepoFile.path)**path** (str) — file path relative to the repo root.
*   [](#huggingface_hub.hf_api.RepoFile.size)**size** (`int`) — The file’s size, in bytes.
*   [](#huggingface_hub.hf_api.RepoFile.blob_id)**blob\_id** (`str`) — The file’s git OID.
*   [](#huggingface_hub.hf_api.RepoFile.lfs)**lfs** (`BlobLfsInfo`) — The file’s LFS metadata.
*   [](#huggingface_hub.hf_api.RepoFile.last_commit)**last\_commit** (`LastCommitInfo`, _optional_) — The file’s last commit metadata. Only defined if [list\_repo\_tree()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree) and [get\_paths\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.get_paths_info) are called with `expand=True`.
*   [](#huggingface_hub.hf_api.RepoFile.security)**security** (`BlobSecurityInfo`, _optional_) — The file’s security scan metadata. Only defined if [list\_repo\_tree()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree) and [get\_paths\_info()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.get_paths_info) are called with `expand=True`.

Contains information about a file on the Hub.

### [](#huggingface_hub.RepoUrl)RepoUrl

### class huggingface\_hub.RepoUrl

[](#huggingface_hub.RepoUrl)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L516)

( url: Anyendpoint: Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.RepoUrl.url)**url** (`Any`) — String value of the repo url.
*   [](#huggingface_hub.RepoUrl.endpoint)**endpoint** (`str`, _optional_) — Endpoint of the Hub. Defaults to [https://huggingface.co](https://huggingface.co).

Raises

export const metadata = 'undefined';

`ValueError`

export const metadata = 'undefined';

*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If URL cannot be parsed.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If `repo_type` is unknown.

Subclass of `str` describing a repo URL on the Hub.

`RepoUrl` is returned by `HfApi.create_repo`. It inherits from `str` for backward compatibility. At initialization, the URL is parsed to populate properties:

*   endpoint (`str`)
*   namespace (`Optional[str]`)
*   repo\_name (`str`)
*   repo\_id (`str`)
*   repo\_type (`Literal["model", "dataset", "space"]`)
*   url (`str`)

[](#huggingface_hub.RepoUrl.example)

Example:

Copied

\>>> RepoUrl('https://huggingface.co/gpt2')
RepoUrl('https://huggingface.co/gpt2', endpoint='https://huggingface.co', repo\_type='model', repo\_id='gpt2')

\>>> RepoUrl('https://hub-ci.huggingface.co/datasets/dummy\_user/dummy\_dataset', endpoint='https://hub-ci.huggingface.co')
RepoUrl('https://hub-ci.huggingface.co/datasets/dummy\_user/dummy\_dataset', endpoint='https://hub-ci.huggingface.co', repo\_type='dataset', repo\_id='dummy\_user/dummy\_dataset')

\>>> RepoUrl('hf://datasets/my-user/my-dataset')
RepoUrl('hf://datasets/my-user/my-dataset', endpoint='https://huggingface.co', repo\_type='dataset', repo\_id='user/dataset')

\>>> HfApi.create\_repo("dummy\_model")
RepoUrl('https://huggingface.co/Wauplin/dummy\_model', endpoint='https://huggingface.co', repo\_type='model', repo\_id='Wauplin/dummy\_model')

### [](#huggingface_hub.utils.SafetensorsRepoMetadata)SafetensorsRepoMetadata

### class huggingface\_hub.utils.SafetensorsRepoMetadata

[](#huggingface_hub.utils.SafetensorsRepoMetadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/utils/_safetensors.py#L73)

( metadata: typing.Optional\[typing.Dict\]sharded: boolweight\_map: typing.Dict\[str, str\]files\_metadata: typing.Dict\[str, huggingface\_hub.utils.\_safetensors.SafetensorsFileMetadata\] )

Parameters

*   [](#huggingface_hub.utils.SafetensorsRepoMetadata.metadata)**metadata** (`Dict`, _optional_) — The metadata contained in the ‘model.safetensors.index.json’ file, if it exists. Only populated for sharded models.
*   [](#huggingface_hub.utils.SafetensorsRepoMetadata.sharded)**sharded** (`bool`) — Whether the repo contains a sharded model or not.
*   [](#huggingface_hub.utils.SafetensorsRepoMetadata.weight_map)**weight\_map** (`Dict[str, str]`) — A map of all weights. Keys are tensor names and values are filenames of the files containing the tensors.
*   [](#huggingface_hub.utils.SafetensorsRepoMetadata.files_metadata)**files\_metadata** (`Dict[str, SafetensorsFileMetadata]`) — A map of all files metadata. Keys are filenames and values are the metadata of the corresponding file, as a `SafetensorsFileMetadata` object.
*   [](#huggingface_hub.utils.SafetensorsRepoMetadata.parameter_count)**parameter\_count** (`Dict[str, int]`) — A map of the number of parameters per data type. Keys are data types and values are the number of parameters of that data type.

Metadata for a Safetensors repo.

A repo is considered to be a Safetensors repo if it contains either a ‘model.safetensors’ weight file (non-shared model) or a ‘model.safetensors.index.json’ index file (sharded model) at its root.

This class is returned by [get\_safetensors\_metadata()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.get_safetensors_metadata).

For more details regarding the safetensors format, check out [https://huggingface.co/docs/safetensors/index#format](https://huggingface.co/docs/safetensors/index#format).

### [](#huggingface_hub.utils.SafetensorsFileMetadata)SafetensorsFileMetadata

### class huggingface\_hub.utils.SafetensorsFileMetadata

[](#huggingface_hub.utils.SafetensorsFileMetadata)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/utils/_safetensors.py#L43)

( metadata: typing.Dict\[str, str\]tensors: typing.Dict\[str, huggingface\_hub.utils.\_safetensors.TensorInfo\] )

Parameters

*   [](#huggingface_hub.utils.SafetensorsFileMetadata.metadata)**metadata** (`Dict`) — The metadata contained in the file.
*   [](#huggingface_hub.utils.SafetensorsFileMetadata.tensors)**tensors** (`Dict[str, TensorInfo]`) — A map of all tensors. Keys are tensor names and values are information about the corresponding tensor, as a `TensorInfo` object.
*   [](#huggingface_hub.utils.SafetensorsFileMetadata.parameter_count)**parameter\_count** (`Dict[str, int]`) — A map of the number of parameters per data type. Keys are data types and values are the number of parameters of that data type.

Metadata for a Safetensors file hosted on the Hub.

This class is returned by [parse\_safetensors\_file\_metadata()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.parse_safetensors_file_metadata).

For more details regarding the safetensors format, check out [https://huggingface.co/docs/safetensors/index#format](https://huggingface.co/docs/safetensors/index#format).

### [](#huggingface_hub.SpaceInfo)SpaceInfo

### class huggingface\_hub.SpaceInfo

[](#huggingface_hub.SpaceInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1018)

( \*\*kwargs )

Expand 19 parameters

Parameters

*   [](#huggingface_hub.SpaceInfo.id)**id** (`str`) — ID of the Space.
*   [](#huggingface_hub.SpaceInfo.author)**author** (`str`, _optional_) — Author of the Space.
*   [](#huggingface_hub.SpaceInfo.sha)**sha** (`str`, _optional_) — Repo SHA at this particular revision.
*   [](#huggingface_hub.SpaceInfo.created_at)**created\_at** (`datetime`, _optional_) — Date of creation of the repo on the Hub. Note that the lowest value is `2022-03-02T23:29:04.000Z`, corresponding to the date when we began to store creation dates.
*   [](#huggingface_hub.SpaceInfo.last_modified)**last\_modified** (`datetime`, _optional_) — Date of last commit to the repo.
*   [](#huggingface_hub.SpaceInfo.private)**private** (`bool`) — Is the repo private.
*   [](#huggingface_hub.SpaceInfo.gated)**gated** (`Literal["auto", "manual", False]`, _optional_) — Is the repo gated. If so, whether there is manual or automatic approval.
*   [](#huggingface_hub.SpaceInfo.disabled)**disabled** (`bool`, _optional_) — Is the Space disabled.
*   [](#huggingface_hub.SpaceInfo.host)**host** (`str`, _optional_) — Host URL of the Space.
*   [](#huggingface_hub.SpaceInfo.subdomain)**subdomain** (`str`, _optional_) — Subdomain of the Space.
*   [](#huggingface_hub.SpaceInfo.likes)**likes** (`int`) — Number of likes of the Space.
*   [](#huggingface_hub.SpaceInfo.tags)**tags** (`List[str]`) — List of tags of the Space.
*   [](#huggingface_hub.SpaceInfo.siblings)**siblings** (`List[RepoSibling]`) — List of [huggingface\_hub.hf\_api.RepoSibling](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.hf_api.RepoSibling) objects that constitute the Space.
*   [](#huggingface_hub.SpaceInfo.card_data)**card\_data** (`SpaceCardData`, _optional_) — Space Card Metadata as a [huggingface\_hub.repocard\_data.SpaceCardData](/docs/huggingface_hub/v0.29.2/en/package_reference/cards#huggingface_hub.SpaceCardData) object.
*   [](#huggingface_hub.SpaceInfo.runtime)**runtime** (`SpaceRuntime`, _optional_) — Space runtime information as a [huggingface\_hub.hf\_api.SpaceRuntime](/docs/huggingface_hub/v0.29.2/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime) object.
*   [](#huggingface_hub.SpaceInfo.sdk)**sdk** (`str`, _optional_) — SDK used by the Space.
*   [](#huggingface_hub.SpaceInfo.models)**models** (`List[str]`, _optional_) — List of models used by the Space.
*   [](#huggingface_hub.SpaceInfo.datasets)**datasets** (`List[str]`, _optional_) — List of datasets used by the Space.
*   [](#huggingface_hub.SpaceInfo.trending_score)**trending\_score** (`int`, _optional_) — Trending score of the Space.

Contains information about a Space on the Hub.

Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made. In general, the more specific the query, the more information is returned. On the contrary, when listing spaces using [list\_spaces()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces) only a subset of the attributes are returned.

### [](#huggingface_hub.utils.TensorInfo)TensorInfo

### class huggingface\_hub.utils.TensorInfo

[](#huggingface_hub.utils.TensorInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/utils/_safetensors.py#L13)

( dtype: typing.Literal\['F64', 'F32', 'F16', 'BF16', 'I64', 'I32', 'I16', 'I8', 'U8', 'BOOL'\]shape: typing.List\[int\]data\_offsets: typing.Tuple\[int, int\] )

Parameters

*   [](#huggingface_hub.utils.TensorInfo.dtype)**dtype** (`str`) — The data type of the tensor (“F64”, “F32”, “F16”, “BF16”, “I64”, “I32”, “I16”, “I8”, “U8”, “BOOL”).
*   [](#huggingface_hub.utils.TensorInfo.shape)**shape** (`List[int]`) — The shape of the tensor.
*   [](#huggingface_hub.utils.TensorInfo.data_offsets)**data\_offsets** (`Tuple[int, int]`) — The offsets of the data in the file as a tuple `[BEGIN, END]`.
*   [](#huggingface_hub.utils.TensorInfo.parameter_count)**parameter\_count** (`int`) — The number of parameters in the tensor.

Information about a tensor.

For more details regarding the safetensors format, check out [https://huggingface.co/docs/safetensors/index#format](https://huggingface.co/docs/safetensors/index#format).

### [](#huggingface_hub.User)User

### class huggingface\_hub.User

[](#huggingface_hub.User)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1379)

( \*\*kwargs )

Expand 16 parameters

Parameters

*   [](#huggingface_hub.User.username)**username** (`str`) — Name of the user on the Hub (unique).
*   [](#huggingface_hub.User.fullname)**fullname** (`str`) — User’s full name.
*   [](#huggingface_hub.User.avatar_url)**avatar\_url** (`str`) — URL of the user’s avatar.
*   [](#huggingface_hub.User.details)**details** (`str`, _optional_) — User’s details.
*   [](#huggingface_hub.User.is_following)**is\_following** (`bool`, _optional_) — Whether the authenticated user is following this user.
*   [](#huggingface_hub.User.is_pro)**is\_pro** (`bool`, _optional_) — Whether the user is a pro user.
*   [](#huggingface_hub.User.num_models)**num\_models** (`int`, _optional_) — Number of models created by the user.
*   [](#huggingface_hub.User.num_datasets)**num\_datasets** (`int`, _optional_) — Number of datasets created by the user.
*   [](#huggingface_hub.User.num_spaces)**num\_spaces** (`int`, _optional_) — Number of spaces created by the user.
*   [](#huggingface_hub.User.num_discussions)**num\_discussions** (`int`, _optional_) — Number of discussions initiated by the user.
*   [](#huggingface_hub.User.num_papers)**num\_papers** (`int`, _optional_) — Number of papers authored by the user.
*   [](#huggingface_hub.User.num_upvotes)**num\_upvotes** (`int`, _optional_) — Number of upvotes received by the user.
*   [](#huggingface_hub.User.num_likes)**num\_likes** (`int`, _optional_) — Number of likes given by the user.
*   [](#huggingface_hub.User.num_following)**num\_following** (`int`, _optional_) — Number of users this user is following.
*   [](#huggingface_hub.User.num_followers)**num\_followers** (`int`, _optional_) — Number of users following this user.
*   [](#huggingface_hub.User.orgs)**orgs** (list of `Organization`) — List of organizations the user is part of.

Contains information about a user on the Hub.

### [](#huggingface_hub.UserLikes)UserLikes

### class huggingface\_hub.UserLikes

[](#huggingface_hub.UserLikes)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L1324)

( user: strtotal: intdatasets: List\[str\]models: List\[str\]spaces: List\[str\] )

Parameters

*   [](#huggingface_hub.UserLikes.user)**user** (`str`) — Name of the user for which we fetched the likes.
*   [](#huggingface_hub.UserLikes.total)**total** (`int`) — Total number of likes.
*   [](#huggingface_hub.UserLikes.datasets)**datasets** (`List[str]`) — List of datasets liked by the user (as repo\_ids).
*   [](#huggingface_hub.UserLikes.models)**models** (`List[str]`) — List of models liked by the user (as repo\_ids).
*   [](#huggingface_hub.UserLikes.spaces)**spaces** (`List[str]`) — List of spaces liked by the user (as repo\_ids).

Contains information about a user likes on the Hub.

### [](#huggingface_hub.WebhookInfo)WebhookInfo

### class huggingface\_hub.WebhookInfo

[](#huggingface_hub.WebhookInfo)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L489)

( id: strurl: strwatched: List\[WebhookWatchedItem\]domains: List\[constants.WEBHOOK\_DOMAIN\_T\]secret: Optional\[str\]disabled: bool )

Parameters

*   [](#huggingface_hub.WebhookInfo.id)**id** (`str`) — ID of the webhook.
*   [](#huggingface_hub.WebhookInfo.url)**url** (`str`) — URL of the webhook.
*   [](#huggingface_hub.WebhookInfo.watched)**watched** (`List[WebhookWatchedItem]`) — List of items watched by the webhook, see [WebhookWatchedItem](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.WebhookWatchedItem).
*   [](#huggingface_hub.WebhookInfo.domains)**domains** (`List[WEBHOOK_DOMAIN_T]`) — List of domains the webhook is watching. Can be one of `["repo", "discussions"]`.
*   [](#huggingface_hub.WebhookInfo.secret)**secret** (`str`, _optional_) — Secret of the webhook.
*   [](#huggingface_hub.WebhookInfo.disabled)**disabled** (`bool`) — Whether the webhook is disabled or not.

Data structure containing information about a webhook.

### [](#huggingface_hub.WebhookWatchedItem)WebhookWatchedItem

### class huggingface\_hub.WebhookWatchedItem

[](#huggingface_hub.WebhookWatchedItem)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/hf_api.py#L474)

( type: Literal\['dataset', 'model', 'org', 'space', 'user'\]name: str )

Parameters

*   [](#huggingface_hub.WebhookWatchedItem.type)**type** (`Literal["dataset", "model", "org", "space", "user"]`) — Type of the item to be watched. Can be one of `["dataset", "model", "org", "space", "user"]`.
*   [](#huggingface_hub.WebhookWatchedItem.name)**name** (`str`) — Name of the item to be watched. Can be the username, organization name, model name, dataset name or space name.

Data structure containing information about the items watched by a webhook.

[](#huggingface_hub.CommitOperationAdd)CommitOperation
------------------------------------------------------

Below are the supported values for `CommitOperation()`:

### class huggingface\_hub.CommitOperationAdd

[](#huggingface_hub.CommitOperationAdd)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_api.py#L118)

( path\_in\_repo: strpath\_or\_fileobj: typing.Union\[str, pathlib.Path, bytes, typing.BinaryIO\] )

Parameters

*   [](#huggingface_hub.CommitOperationAdd.path_in_repo)**path\_in\_repo** (`str`) — Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
*   [](#huggingface_hub.CommitOperationAdd.path_or_fileobj)**path\_or\_fileobj** (`str`, `Path`, `bytes`, or `BinaryIO`) — Either:
    
    *   a path to a local file (as `str` or `pathlib.Path`) to upload
    *   a buffer of bytes (`bytes`) holding the content of the file to upload
    *   a “file object” (subclass of `io.BufferedIOBase`), typically obtained with `open(path, "rb")`. It must support `seek()` and `tell()` methods.
    

Raises

export const metadata = 'undefined';

`ValueError`

export const metadata = 'undefined';

*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If `path_or_fileobj` is not one of `str`, `Path`, `bytes` or `io.BufferedIOBase`.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If `path_or_fileobj` is a `str` or `Path` but not a path to an existing file.
*   [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) — If `path_or_fileobj` is a `io.BufferedIOBase` but it doesn’t support both `seek()` and `tell()`.

Data structure holding necessary info to upload a file to a repository on the Hub.

#### as\_file

[](#huggingface_hub.CommitOperationAdd.as_file)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_api.py#L201)

( with\_tqdm: bool = False )

Parameters

*   [](#huggingface_hub.CommitOperationAdd.as_file.with_tqdm)**with\_tqdm** (`bool`, _optional_, defaults to `False`) — If True, iterating over the file object will display a progress bar. Only works if the file-like object is a path to a file. Pure bytes and buffers are not supported.

A context manager that yields a file-like object allowing to read the underlying data behind `path_or_fileobj`.

[](#huggingface_hub.CommitOperationAdd.as_file.example)

Example:

Copied

\>>> operation = CommitOperationAdd(
...        path\_in\_repo="remote/dir/weights.h5",
...        path\_or\_fileobj="./local/weights.h5",
... )
CommitOperationAdd(path\_in\_repo='remote/dir/weights.h5', path\_or\_fileobj='./local/weights.h5')

\>>> with operation.as\_file() as file:
...     content = file.read()

\>>> with operation.as\_file(with\_tqdm=True) as file:
...     while True:
...         data = file.read(1024)
...         if not data:
...              break
config.json: 100%|█████████████████████████| 8.19k/8.19k \[00:02<00:00, 3.72kB/s\]

\>>> with operation.as\_file(with\_tqdm=True) as file:
...     requests.put(..., data=file)
config.json: 100%|█████████████████████████| 8.19k/8.19k \[00:02<00:00, 3.72kB/s\]

#### b64content

[](#huggingface_hub.CommitOperationAdd.b64content)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_api.py#L251)

( )

The base64-encoded content of `path_or_fileobj`

Returns: `bytes`

### class huggingface\_hub.CommitOperationDelete

[](#huggingface_hub.CommitOperationDelete)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_api.py#L51)

( path\_in\_repo: stris\_folder: typing.Union\[bool, typing.Literal\['auto'\]\] = 'auto' )

Parameters

*   [](#huggingface_hub.CommitOperationDelete.path_in_repo)**path\_in\_repo** (`str`) — Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"` for a file or `"checkpoints/1fec34a/"` for a folder.
*   [](#huggingface_hub.CommitOperationDelete.is_folder)**is\_folder** (`bool` or `Literal["auto"]`, _optional_) — Whether the Delete Operation applies to a folder or not. If “auto”, the path type (file or folder) is guessed automatically by looking if path ends with a ”/” (folder) or not (file). To explicitly set the path type, you can set `is_folder=True` or `is_folder=False`.

Data structure holding necessary info to delete a file or a folder from a repository on the Hub.

### class huggingface\_hub.CommitOperationCopy

[](#huggingface_hub.CommitOperationCopy)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_api.py#L82)

( src\_path\_in\_repo: strpath\_in\_repo: strsrc\_revision: typing.Optional\[str\] = None\_src\_oid: typing.Optional\[str\] = None\_dest\_oid: typing.Optional\[str\] = None )

Parameters

*   [](#huggingface_hub.CommitOperationCopy.src_path_in_repo)**src\_path\_in\_repo** (`str`) — Relative filepath in the repo of the file to be copied, e.g. `"checkpoints/1fec34a/weights.bin"`.
*   [](#huggingface_hub.CommitOperationCopy.path_in_repo)**path\_in\_repo** (`str`) — Relative filepath in the repo where to copy the file, e.g. `"checkpoints/1fec34a/weights_copy.bin"`.
*   [](#huggingface_hub.CommitOperationCopy.src_revision)**src\_revision** (`str`, _optional_) — The git revision of the file to be copied. Can be any valid git revision. Default to the target commit revision.

Data structure holding necessary info to copy a file in a repository on the Hub.

Limitations:

*   Only LFS files can be copied. To copy a regular file, you need to download it locally and re-upload it
*   Cross-repository copies are not supported.

Note: you can combine a [CommitOperationCopy](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy) and a [CommitOperationDelete](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitOperationDelete) to rename an LFS file on the Hub.

[](#huggingface_hub.CommitScheduler)CommitScheduler
---------------------------------------------------

### class huggingface\_hub.CommitScheduler

[](#huggingface_hub.CommitScheduler)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_scheduler.py#L29)

( repo\_id: strfolder\_path: typing.Union\[str, pathlib.Path\]every: typing.Union\[int, float\] = 5path\_in\_repo: typing.Optional\[str\] = Nonerepo\_type: typing.Optional\[str\] = Nonerevision: typing.Optional\[str\] = Noneprivate: typing.Optional\[bool\] = Nonetoken: typing.Optional\[str\] = Noneallow\_patterns: typing.Union\[typing.List\[str\], str, NoneType\] = Noneignore\_patterns: typing.Union\[typing.List\[str\], str, NoneType\] = Nonesquash\_history: bool = Falsehf\_api: typing.Optional\[ForwardRef('HfApi')\] = None )

Expand 12 parameters

Parameters

*   [](#huggingface_hub.CommitScheduler.repo_id)**repo\_id** (`str`) — The id of the repo to commit to.
*   [](#huggingface_hub.CommitScheduler.folder_path)**folder\_path** (`str` or `Path`) — Path to the local folder to upload regularly.
*   [](#huggingface_hub.CommitScheduler.every)**every** (`int` or `float`, _optional_) — The number of minutes between each commit. Defaults to 5 minutes.
*   [](#huggingface_hub.CommitScheduler.path_in_repo)**path\_in\_repo** (`str`, _optional_) — Relative path of the directory in the repo, for example: `"checkpoints/"`. Defaults to the root folder of the repository.
*   [](#huggingface_hub.CommitScheduler.repo_type)**repo\_type** (`str`, _optional_) — The type of the repo to commit to. Defaults to `model`.
*   [](#huggingface_hub.CommitScheduler.revision)**revision** (`str`, _optional_) — The revision of the repo to commit to. Defaults to `main`.
*   [](#huggingface_hub.CommitScheduler.private)**private** (`bool`, _optional_) — Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
*   [](#huggingface_hub.CommitScheduler.token)**token** (`str`, _optional_) — The token to use to commit to the repo. Defaults to the token saved on the machine.
*   [](#huggingface_hub.CommitScheduler.allow_patterns)**allow\_patterns** (`List[str]` or `str`, _optional_) — If provided, only files matching at least one pattern are uploaded.
*   [](#huggingface_hub.CommitScheduler.ignore_patterns)**ignore\_patterns** (`List[str]` or `str`, _optional_) — If provided, files matching any of the patterns are not uploaded.
*   [](#huggingface_hub.CommitScheduler.squash_history)**squash\_history** (`bool`, _optional_) — Whether to squash the history of the repo after each commit. Defaults to `False`. Squashing commits is useful to avoid degraded performances on the repo when it grows too large.
*   [](#huggingface_hub.CommitScheduler.hf_api)**hf\_api** (`HfApi`, _optional_) — The [HfApi](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi) client to use to commit to the Hub. Can be set with custom settings (user agent, token,…).

Scheduler to upload a local folder to the Hub at regular intervals (e.g. push to hub every 5 minutes).

The recommended way to use the scheduler is to use it as a context manager. This ensures that the scheduler is properly stopped and the last commit is triggered when the script ends. The scheduler can also be stopped manually with the `stop` method. Checkout the [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#scheduled-uploads) to learn more about how to use it.

[](#huggingface_hub.CommitScheduler.example)

Example:

Copied

\>>> from pathlib import Path
\>>> from huggingface\_hub import CommitScheduler

\# Scheduler uploads every 10 minutes
\>>> csv\_path = Path("watched\_folder/data.csv")
\>>> CommitScheduler(repo\_id="test\_scheduler", repo\_type="dataset", folder\_path=csv\_path.parent, every=10)

\>>> with csv\_path.open("a") as f:
...     f.write("first line")

\# Some time later (...)
\>>> with csv\_path.open("a") as f:
...     f.write("second line")

[](#huggingface_hub.CommitScheduler.example-2)

Example using a context manager:

Copied

\>>> from pathlib import Path
\>>> from huggingface\_hub import CommitScheduler

\>>> with CommitScheduler(repo\_id="test\_scheduler", repo\_type="dataset", folder\_path="watched\_folder", every=10) as scheduler:
...     csv\_path = Path("watched\_folder/data.csv")
...     with csv\_path.open("a") as f:
...         f.write("first line")
...     (...)
...     with csv\_path.open("a") as f:
...         f.write("second line")

\# Scheduler is now stopped and last commit have been triggered

#### push\_to\_hub

[](#huggingface_hub.CommitScheduler.push_to_hub)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_scheduler.py#L204)

( )

Push folder to the Hub and return the commit info.

This method is not meant to be called directly. It is run in the background by the scheduler, respecting a queue mechanism to avoid concurrent commits. Making a direct call to the method might lead to concurrency issues.

The default behavior of `push_to_hub` is to assume an append-only folder. It lists all files in the folder and uploads only changed files. If no changes are found, the method returns without committing anything. If you want to change this behavior, you can inherit from [CommitScheduler](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.CommitScheduler) and override this method. This can be useful for example to compress data together in a single file before committing. For more details and examples, check out our [integration guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#scheduled-uploads).

#### stop

[](#huggingface_hub.CommitScheduler.stop)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_scheduler.py#L157)

( )

Stop the scheduler.

A stopped scheduler cannot be restarted. Mostly for tests purposes.

#### trigger

[](#huggingface_hub.CommitScheduler.trigger)[< source \>](https://github.com/huggingface/huggingface_hub/blob/v0.29.2/src/huggingface_hub/_commit_scheduler.py#L181)

( )

Trigger a `push_to_hub` and return a future.

This method is automatically called every `every` minutes. You can also call it manually to trigger a commit immediately, without waiting for the next scheduled commit.

[< \> Update on GitHub](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/package_reference/hf_api.md)

[←Managing local and online repositories](/docs/huggingface_hub/en/package_reference/repository) [Downloading files→](/docs/huggingface_hub/en/package_reference/file_download)