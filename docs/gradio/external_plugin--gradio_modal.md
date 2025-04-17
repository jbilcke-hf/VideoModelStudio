Description du projet
---------------------

`gradio_modal`
==============

[![PyPI - Version](https://pypi-camo.freetls.fastly.net/19d01702f9691477566e07fbd3c8eb08188e6eae/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f67726164696f5f6d6f64616c)](https://pypi.org/project/gradio_modal/)

A popup modal component

Installation
------------

pip install gradio\_modal

Usage
-----

import gradio as gr
from gradio\_modal import Modal

with gr.Blocks() as demo:
    with gr.Tab("Tab 1"):
        text\_1 \= gr.Textbox(label\="Input 1")
        text\_2 \= gr.Textbox(label\="Input 2")
        text\_1.submit(lambda x:x, text\_1, text\_2)
        show\_btn \= gr.Button("Show Modal")
        show\_btn2 \= gr.Button("Show Modal 2")
        gr.Examples(
            \[\["Text 1", "Text 2"\], \["Text 3", "Text 4"\]\],
            inputs\=\[text\_1, text\_2\],
        )
    with gr.Tab("Tab 2"):
        gr.Markdown("This is tab 2")
    with Modal(visible\=False) as modal:
        for i in range(5):
            gr.Markdown("Hello world!")
    with Modal(visible\=False) as modal2:
        for i in range(100):
            gr.Markdown("Hello world!")
    show\_btn.click(lambda: Modal(visible\=True), None, modal)
    show\_btn2.click(lambda: Modal(visible\=True), None, modal2)

if \_\_name\_\_ \== "\_\_main\_\_":
    demo.launch()

`Modal`
-------

### Initialization

name

type

default

description

`visible`

bool

`False`

If False, modal will be hidden.

`elem_id`

str | None

`None`

An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.

`elem_classes`

list\[str\] | str | None

`None`

An optional string or list of strings that are assigned as the class of this component in the HTML DOM. Can be used for targeting CSS styles.

`allow_user_close`

bool

`True`

If True, user can close the modal (by clicking outside, clicking the X, or the escape key).

`render`

bool

`True`

If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.

### Events

name

description

`blur`

This listener is triggered when the Modal is unfocused/blurred.