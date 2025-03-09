[](#load-a-dataset-from-the-hub)Load a dataset from the Hub
===========================================================

Finding high-quality datasets that are reproducible and accessible can be difficult. One of 🤗 Datasets main goals is to provide a simple way to load a dataset of any format or type. The easiest way to get started is to discover an existing dataset on the [Hugging Face Hub](https://huggingface.co/datasets) - a community-driven collection of datasets for tasks in NLP, computer vision, and audio - and use 🤗 Datasets to download and generate the dataset.

This tutorial uses the [rotten\_tomatoes](https://huggingface.co/datasets/rotten_tomatoes) and [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) datasets, but feel free to load any dataset you want and follow along. Head over to the Hub now and find a dataset for your task!

[](#load-a-dataset)Load a dataset
---------------------------------

Before you take the time to download a dataset, it’s often helpful to quickly get some general information about a dataset. A dataset’s information is stored inside [DatasetInfo](/docs/datasets/v3.3.2/en/package_reference/main_classes#datasets.DatasetInfo) and can include information such as the dataset description, features, and dataset size.

Use the [load\_dataset\_builder()](/docs/datasets/v3.3.2/en/package_reference/loading_methods#datasets.load_dataset_builder) function to load a dataset builder and inspect a dataset’s attributes without committing to downloading it:

Copied

\>>> from datasets import load\_dataset\_builder
\>>> ds\_builder = load\_dataset\_builder("cornell-movie-review-data/rotten\_tomatoes")

\# Inspect dataset description
\>>> ds\_builder.info.description
Movie Review Dataset. This is a dataset of containing 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. This data was first used in Bo Pang and Lillian Lee, \`\`Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.'', Proceedings of the ACL, 2005.

\# Inspect dataset features
\>>> ds\_builder.info.features
{'label': ClassLabel(names=\['neg', 'pos'\], id\=None),
 'text': Value(dtype='string', id\=None)}

If you’re happy with the dataset, then load it with [load\_dataset()](/docs/datasets/v3.3.2/en/package_reference/loading_methods#datasets.load_dataset):

Copied

\>>> from datasets import load\_dataset

\>>> dataset = load\_dataset("cornell-movie-review-data/rotten\_tomatoes", split="train")

[](#splits)Splits
-----------------

A split is a specific subset of a dataset like `train` and `test`. List a dataset’s split names with the [get\_dataset\_split\_names()](/docs/datasets/v3.3.2/en/package_reference/loading_methods#datasets.get_dataset_split_names) function:

Copied

\>>> from datasets import get\_dataset\_split\_names

\>>> get\_dataset\_split\_names("cornell-movie-review-data/rotten\_tomatoes")
\['train', 'validation', 'test'\]

Then you can load a specific split with the `split` parameter. Loading a dataset `split` returns a [Dataset](/docs/datasets/v3.3.2/en/package_reference/main_classes#datasets.Dataset) object:

Copied

\>>> from datasets import load\_dataset

\>>> dataset = load\_dataset("cornell-movie-review-data/rotten\_tomatoes", split="train")
\>>> dataset
Dataset({
    features: \['text', 'label'\],
    num\_rows: 8530
})

If you don’t specify a `split`, 🤗 Datasets returns a [DatasetDict](/docs/datasets/v3.3.2/en/package_reference/main_classes#datasets.DatasetDict) object instead:

Copied

\>>> from datasets import load\_dataset

\>>> dataset = load\_dataset("cornell-movie-review-data/rotten\_tomatoes")
DatasetDict({
    train: Dataset({
        features: \['text', 'label'\],
        num\_rows: 8530
    })
    validation: Dataset({
        features: \['text', 'label'\],
        num\_rows: 1066
    })
    test: Dataset({
        features: \['text', 'label'\],
        num\_rows: 1066
    })
})

[](#configurations)Configurations
---------------------------------

Some datasets contain several sub-datasets. For example, the [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) dataset has several sub-datasets, each one containing audio data in a different language. These sub-datasets are known as _configurations_ or _subsets_, and you must explicitly select one when loading the dataset. If you don’t provide a configuration name, 🤗 Datasets will raise a `ValueError` and remind you to choose a configuration.

Use the [get\_dataset\_config\_names()](/docs/datasets/v3.3.2/en/package_reference/loading_methods#datasets.get_dataset_config_names) function to retrieve a list of all the possible configurations available to your dataset:

Copied

\>>> from datasets import get\_dataset\_config\_names

\>>> configs = get\_dataset\_config\_names("PolyAI/minds14")
\>>> print(configs)
\['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN', 'all'\]

Then load the configuration you want:

Copied

\>>> from datasets import load\_dataset

\>>> mindsFR = load\_dataset("PolyAI/minds14", "fr-FR", split="train")

[](#remote-code)Remote code
---------------------------

Certain datasets repositories contain a loading script with the Python code used to generate the dataset. All files and code uploaded to the Hub are scanned for malware (refer to the Hub security documentation for more information), but you should still review the dataset loading scripts and authors to avoid executing malicious code on your machine. You should set `trust_remote_code=True` to use a dataset with a loading script, or you will get an error:

Copied

\>>> from datasets import get\_dataset\_config\_names, get\_dataset\_split\_names, load\_dataset

\>>> c4 = load\_dataset("c4", "en", split="train", trust\_remote\_code=True)
\>>> get\_dataset\_config\_names("c4", trust\_remote\_code=True)
\['en', 'realnewslike', 'en.noblocklist', 'en.noclean'\]
\>>> get\_dataset\_split\_names("c4", "en", trust\_remote\_code=True)
\['train', 'validation'\]

For security reasons, 🤗 Datasets do not allow running dataset loading scripts by default, and you have to pass `trust_remote_code=True` to load datasets that require running a dataset script.

[< \> Update on GitHub](https://github.com/huggingface/datasets/blob/main/docs/source/load_hub.mdx)

[←Overview](/docs/datasets/en/tutorial) [Know your dataset→](/docs/datasets/en/access)