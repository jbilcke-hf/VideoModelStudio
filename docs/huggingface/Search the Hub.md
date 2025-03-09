[](#search-the-hub)Search the Hub
=================================

In this tutorial, you will learn how to search models, datasets and spaces on the Hub using `huggingface_hub`.

[](#how-to-list-repositories-)How to list repositories ?
--------------------------------------------------------

`huggingface_hub` library includes an HTTP client [HfApi](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi) to interact with the Hub. Among other things, it can list models, datasets and spaces stored on the Hub:

Copied

\>>> from huggingface\_hub import HfApi
\>>> api = HfApi()
\>>> models = api.list\_models()

The output of [list\_models()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_models) is an iterator over the models stored on the Hub.

Similarly, you can use [list\_datasets()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) to list datasets and [list\_spaces()](/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces) to list Spaces.

[](#how-to-filter-repositories-)How to filter repositories ?
------------------------------------------------------------

Listing repositories is great but now you might want to filter your search. The list helpers have several attributes like:

*   `filter`
*   `author`
*   `search`
*   …

Let’s see an example to get all models on the Hub that does image classification, have been trained on the imagenet dataset and that runs with PyTorch.

Copied

models = hf\_api.list\_models(
	task="image-classification",
	library="pytorch",
	trained\_dataset="imagenet",
)

While filtering, you can also sort the models and take only the top results. For example, the following example fetches the top 5 most downloaded datasets on the Hub:

Copied

\>>> list(list\_datasets(sort="downloads", direction=-1, limit=5))
\[DatasetInfo(
	id\='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last\_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)

To explore available filters on the Hub, visit [models](https://huggingface.co/models) and [datasets](https://huggingface.co/datasets) pages in your browser, search for some parameters and look at the values in the URL.

[< \> Update on GitHub](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/search.md)

HfApi Client

[←Repository](/docs/huggingface_hub/en/guides/repository) [Inference→](/docs/huggingface_hub/en/guides/inference)