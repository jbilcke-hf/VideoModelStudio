1.  Other Tutorials
2.  Theming Guide

[

←

Styling The Gradio Dataframe



](../guides/styling-the-gradio-dataframe/)[

Understanding Gradio Share Links

→

](../guides/understanding-gradio-share-links/)

.wrapper { position: relative; padding-bottom: 56.25%; padding-top: 25px; height: 0; } .wrapper iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

Theming
=======

Introduction[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#introduction)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Gradio features a built-in theming engine that lets you customize the look and feel of your app. You can choose from a variety of themes, or create your own. To do so, pass the `theme=` kwarg to the `Blocks` or `Interface` constructor. For example:

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        ...

Gradio comes with a set of prebuilt themes which you can load from `gr.themes.*`. These are:

*   `gr.themes.Base()` - the `"base"` theme sets the primary color to blue but otherwise has minimal styling, making it particularly useful as a base for creating new, custom themes.
*   `gr.themes.Default()` - the `"default"` Gradio 5 theme, with a vibrant orange primary color and gray secondary color.
*   `gr.themes.Origin()` - the `"origin"` theme is most similar to Gradio 4 styling. Colors, especially in light mode, are more subdued than the Gradio 5 default theme.
*   `gr.themes.Citrus()` - the `"citrus"` theme uses a yellow primary color, highlights form elements that are in focus, and includes fun 3D effects when buttons are clicked.
*   `gr.themes.Monochrome()` - the `"monochrome"` theme uses a black primary and white secondary color, and uses serif-style fonts, giving the appearance of a black-and-white newspaper.
*   `gr.themes.Soft()` - the `"soft"` theme uses a purple primary color and white secondary color. It also increases the border radius around buttons and form elements and highlights labels.
*   `gr.themes.Glass()` - the `"glass"` theme has a blue primary color and a transclucent gray secondary color. The theme also uses vertical gradients to create a glassy effect.
*   `gr.themes.Ocean()` - the `"ocean"` theme has a blue-green primary color and gray secondary color. The theme also uses horizontal gradients, especially for buttons and some form elements.

Each of these themes set values for hundreds of CSS variables. You can use prebuilt themes as a starting point for your own custom themes, or you can create your own themes from scratch. Let's take a look at each approach.

Using the Theme Builder[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#using-the-theme-builder)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The easiest way to build a theme is using the Theme Builder. To launch the Theme Builder locally, run the following code:

    import gradio as gr
    
    gr.themes.builder()

Undo Dark Mode

Source Theme Core Colors Core Sizing Core Fonts Body Attributes Element Colors Text Shadows Layout Atoms Component Atoms Buttons

Source Theme Core Colors Core Sizing Core Fonts Body Attributes

Element Colors Text Shadows Layout Atoms Component Atoms Buttons

Select a base theme below you would like to build off of. Note: when you click 'Load Theme', all variable values in other tabs will be overwritten!

Theme

Load Theme

Set the three hues of the theme: `primary_hue`, `secondary_hue`, and `neutral_hue`. Each of these is a palette ranging from 50 to 950 in brightness. Pick a preset palette - optionally, open the accordion to overwrite specific values. Note that these variables do not affect elements directly, but are referenced by other variables with asterisks, such as `*primary_200` or `*neutral_950`.

Primary Hue

Primary Hue Palette ▼

primary\_50

primary\_100

primary\_200

primary\_300

primary\_400

primary\_500

primary\_600

primary\_700

primary\_800

primary\_900

primary\_950

Secondary Hue

Secondary Hue Palette ▼

secondary\_50

secondary\_100

secondary\_200

secondary\_300

secondary\_400

secondary\_500

secondary\_600

secondary\_700

secondary\_800

secondary\_900

secondary\_950

Neutral hue

Neutral Hue Palette ▼

neutral\_50

neutral\_100

neutral\_200

neutral\_300

neutral\_400

neutral\_500

neutral\_600

neutral\_700

neutral\_800

neutral\_900

neutral\_950

Set the sizing of the theme via: `text_size`, `spacing_size`, and `radius_size`. Each of these is set to a collection of sizes ranging from `xxs` to `xxl`. Pick a preset size collection - optionally, open the accordion to overwrite specific values. Note that these variables do not affect elements directly, but are referenced by other variables with asterisks, such as `*spacing_xl` or `*text_sm`.

Text Size

Text Size Range ▼

text\_xxs

text\_xs

text\_sm

text\_md

text\_lg

text\_xl

text\_xxl

Spacing Size

Spacing Size Range ▼

spacing\_xxs

spacing\_xs

spacing\_sm

spacing\_md

spacing\_lg

spacing\_xl

spacing\_xxl

Radius Size

Radius Size Range ▼

radius\_xxs

radius\_xs

radius\_sm

radius\_md

radius\_lg

radius\_xl

radius\_xxl

Set the main `font` and the monospace `font_mono` here. Set up to 4 values for each (fallbacks in case a font is not available). Check "Google Font" if font should be loaded from Google Fonts.

### Main Font

Font 1

 Google Font

Font 2

 Google Font

Font 3

 Google Font

Font 4

 Google Font

### Monospace Font

Font 1

 Google Font

Font 2

 Google Font

Font 3

 Google Font

Font 4

 Google Font

These set set the values for the entire body of the app. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

body\_background\_fill

The background of the entire app.

body\_background\_fill\_dark

The background of the entire app in dark mode.

body\_text\_color

The default text color.

body\_text\_color\_dark

The default text color in dark mode.

body\_text\_size

The default text size.

body\_text\_color\_subdued

The text color used for softer, less important text.

body\_text\_color\_subdued\_dark

The text color used for softer, less important text in dark mode.

body\_text\_weight

The default text weight.

embed\_radius

The corner radius used for embedding when the app is embedded within a page.

These set the colors for common elements. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

background\_fill\_primary

The background primarily used for items placed directly on the page.

background\_fill\_primary\_dark

The background primarily used for items placed directly on the page in dark mode.

background\_fill\_secondary

The background primarily used for items placed on top of another item.

background\_fill\_secondary\_dark

The background primarily used for items placed on top of another item in dark mode.

border\_color\_accent

The border color used for accented items.

border\_color\_accent\_dark

The border color used for accented items in dark mode.

border\_color\_accent\_subdued

The subdued border color for accented items.

border\_color\_accent\_subdued\_dark

The subdued border color for accented items in dark mode.

border\_color\_primary

The border color primarily used for items placed directly on the page.

border\_color\_primary\_dark

The border color primarily used for items placed directly on the page in dark mode.

color\_accent

The color used for accented items.

color\_accent\_soft

The softer color used for accented items.

color\_accent\_soft\_dark

The softer color used for accented items in dark mode.

This sets the text styling for text elements. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

link\_text\_color

The text color used for links.

link\_text\_color\_dark

The text color used for links in dark mode.

link\_text\_color\_active

The text color used for links when they are active.

link\_text\_color\_active\_dark

The text color used for links when they are active in dark mode.

link\_text\_color\_hover

The text color used for links when they are hovered over.

link\_text\_color\_hover\_dark

The text color used for links when they are hovered over in dark mode.

link\_text\_color\_visited

The text color used for links when they have been visited.

link\_text\_color\_visited\_dark

The text color used for links when they have been visited in dark mode.

prose\_text\_size

The text size used for markdown and other prose.

prose\_text\_weight

The text weight used for markdown and other prose.

prose\_header\_text\_weight

The text weight of a header used for markdown and other prose.

code\_background\_fill

The background color of code blocks.

code\_background\_fill\_dark

The background color of code blocks in dark mode.

These set the high-level shadow rendering styles. These variables are often referenced by other component-specific shadow variables. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

shadow\_drop

Drop shadow used by other shadowed items.

shadow\_drop\_lg

Larger drop shadow used by other shadowed items.

shadow\_inset

Inset shadow used by other shadowed items.

shadow\_spread

Size of shadow spread used by shadowed items.

shadow\_spread\_dark

Size of shadow spread used by shadowed items in dark mode.

These set the style for common layout elements, such as the blocks that wrap components. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

block\_background\_fill

The background around an item.

block\_background\_fill\_dark

The background around an item in dark mode.

block\_border\_color

The border color around an item.

block\_border\_color\_dark

The border color around an item in dark mode.

block\_border\_width

The border width around an item.

block\_border\_width\_dark

The border width around an item in dark mode.

block\_info\_text\_color

The color of the info text.

block\_info\_text\_color\_dark

The color of the info text in dark mode.

block\_info\_text\_size

The size of the info text.

block\_info\_text\_weight

The weight of the info text.

block\_label\_background\_fill

The background of the title label of a media element (e.g. image).

block\_label\_background\_fill\_dark

The background of the title label of a media element (e.g. image) in dark mode.

block\_label\_border\_color

The border color of the title label of a media element (e.g. image).

block\_label\_border\_color\_dark

The border color of the title label of a media element (e.g. image) in dark mode.

block\_label\_border\_width

The border width of the title label of a media element (e.g. image).

block\_label\_border\_width\_dark

The border width of the title label of a media element (e.g. image) in dark mode.

block\_label\_shadow

The shadow of the title label of a media element (e.g. image).

block\_label\_text\_color

The text color of the title label of a media element (e.g. image).

block\_label\_text\_color\_dark

The text color of the title label of a media element (e.g. image) in dark mode.

block\_label\_margin

The margin of the title label of a media element (e.g. image) from its surrounding container.

block\_label\_padding

The padding of the title label of a media element (e.g. image).

block\_label\_radius

The corner radius of the title label of a media element (e.g. image).

block\_label\_right\_radius

The corner radius of a right-aligned helper label.

block\_label\_text\_size

The text size of the title label of a media element (e.g. image).

block\_label\_text\_weight

The text weight of the title label of a media element (e.g. image).

block\_padding

The padding around an item.

block\_radius

The corner radius around an item.

block\_shadow

The shadow under an item.

block\_shadow\_dark

The shadow under an item in dark mode.

block\_title\_background\_fill

The background of the title of a form element (e.g. textbox).

block\_title\_background\_fill\_dark

The background of the title of a form element (e.g. textbox) in dark mode.

block\_title\_border\_color

The border color of the title of a form element (e.g. textbox).

block\_title\_border\_color\_dark

The border color of the title of a form element (e.g. textbox) in dark mode.

block\_title\_border\_width

The border width of the title of a form element (e.g. textbox).

block\_title\_border\_width\_dark

The border width of the title of a form element (e.g. textbox) in dark mode.

block\_title\_text\_color

The text color of the title of a form element (e.g. textbox).

block\_title\_text\_color\_dark

The text color of the title of a form element (e.g. textbox) in dark mode.

block\_title\_padding

The padding of the title of a form element (e.g. textbox).

block\_title\_radius

The corner radius of the title of a form element (e.g. textbox).

block\_title\_text\_size

The text size of the title of a form element (e.g. textbox).

block\_title\_text\_weight

The text weight of the title of a form element (e.g. textbox).

container\_radius

The corner radius of a layout component that holds other content.

form\_gap\_width

The border gap between form elements, (e.g. consecutive textboxes).

layout\_gap

The gap between items within a row or column.

panel\_background\_fill

The background of a panel.

panel\_background\_fill\_dark

The background of a panel in dark mode.

panel\_border\_color

The border color of a panel.

panel\_border\_color\_dark

The border color of a panel in dark mode.

panel\_border\_width

The border width of a panel.

panel\_border\_width\_dark

The border width of a panel in dark mode.

section\_header\_text\_size

The text size of a section header (e.g. tab name).

section\_header\_text\_weight

The text weight of a section header (e.g. tab name).

These set the style for elements within components. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

accordion\_text\_color

The body text color in the accordion.

accordion\_text\_color\_dark

The body text color in the accordion in dark mode.

table\_text\_color

The body text color in the table.

table\_text\_color\_dark

The body text color in the table in dark mode.

checkbox\_background\_color

The background of a checkbox square or radio circle.

chatbot\_text\_size

The text size of the chatbot text.

checkbox\_background\_color\_dark

The background of a checkbox square or radio circle in dark mode.

checkbox\_background\_color\_focus

The background of a checkbox square or radio circle when focused.

checkbox\_background\_color\_focus\_dark

The background of a checkbox square or radio circle when focused in dark mode.

checkbox\_background\_color\_hover

The background of a checkbox square or radio circle when hovered over.

checkbox\_background\_color\_hover\_dark

The background of a checkbox square or radio circle when hovered over in dark mode.

checkbox\_background\_color\_selected

The background of a checkbox square or radio circle when selected.

checkbox\_background\_color\_selected\_dark

The background of a checkbox square or radio circle when selected in dark mode.

checkbox\_border\_color

The border color of a checkbox square or radio circle.

checkbox\_border\_color\_dark

The border color of a checkbox square or radio circle in dark mode.

checkbox\_border\_color\_focus

The border color of a checkbox square or radio circle when focused.

checkbox\_border\_color\_focus\_dark

The border color of a checkbox square or radio circle when focused in dark mode.

checkbox\_border\_color\_hover

The border color of a checkbox square or radio circle when hovered over.

checkbox\_border\_color\_hover\_dark

The border color of a checkbox square or radio circle when hovered over in dark mode.

checkbox\_border\_color\_selected

The border color of a checkbox square or radio circle when selected.

checkbox\_border\_color\_selected\_dark

The border color of a checkbox square or radio circle when selected in dark mode.

checkbox\_border\_radius

The corner radius of a checkbox square.

checkbox\_border\_width

The border width of a checkbox square or radio circle.

checkbox\_border\_width\_dark

The border width of a checkbox square or radio circle in dark mode.

checkbox\_check

The checkmark visual of a checkbox square.

radio\_circle

The circle visual of a radio circle.

checkbox\_shadow

The shadow of a checkbox square or radio circle.

checkbox\_label\_background\_fill

The background of the surrounding button of a checkbox or radio element.

checkbox\_label\_background\_fill\_dark

The background of the surrounding button of a checkbox or radio element in dark mode.

checkbox\_label\_background\_fill\_hover

The background of the surrounding button of a checkbox or radio element when hovered over.

checkbox\_label\_background\_fill\_hover\_dark

The background of the surrounding button of a checkbox or radio element when hovered over in dark mode.

checkbox\_label\_background\_fill\_selected

The background of the surrounding button of a checkbox or radio element when selected.

checkbox\_label\_background\_fill\_selected\_dark

The background of the surrounding button of a checkbox or radio element when selected in dark mode.

checkbox\_label\_border\_color

The border color of the surrounding button of a checkbox or radio element.

checkbox\_label\_border\_color\_dark

The border color of the surrounding button of a checkbox or radio element in dark mode.

checkbox\_label\_border\_color\_hover

The border color of the surrounding button of a checkbox or radio element when hovered over.

checkbox\_label\_border\_color\_hover\_dark

The border color of the surrounding button of a checkbox or radio element when hovered over in dark mode.

checkbox\_label\_border\_color\_selected

The border color of the surrounding button of a checkbox or radio element when selected.

checkbox\_label\_border\_color\_selected\_dark

The border color of the surrounding button of a checkbox or radio element when selected in dark mode.

checkbox\_label\_border\_width

The border width of the surrounding button of a checkbox or radio element.

checkbox\_label\_border\_width\_dark

The border width of the surrounding button of a checkbox or radio element in dark mode.

checkbox\_label\_gap

The gap consecutive checkbox or radio elements.

checkbox\_label\_padding

The padding of the surrounding button of a checkbox or radio element.

checkbox\_label\_shadow

The shadow of the surrounding button of a checkbox or radio element.

checkbox\_label\_text\_size

The text size of the label accompanying a checkbox or radio element.

checkbox\_label\_text\_weight

The text weight of the label accompanying a checkbox or radio element.

checkbox\_label\_text\_color

The text color of the label accompanying a checkbox or radio element.

checkbox\_label\_text\_color\_dark

The text color of the label accompanying a checkbox or radio element in dark mode.

checkbox\_label\_text\_color\_selected

The text color of the label accompanying a checkbox or radio element when selected.

checkbox\_label\_text\_color\_selected\_dark

The text color of the label accompanying a checkbox or radio element when selected in dark mode.

error\_background\_fill

The background of an error message.

error\_background\_fill\_dark

The background of an error message in dark mode.

error\_border\_color

The border color of an error message.

error\_border\_color\_dark

The border color of an error message in dark mode.

error\_border\_width

The border width of an error message.

error\_border\_width\_dark

The border width of an error message in dark mode.

error\_text\_color

The text color of an error message.

error\_text\_color\_dark

The text color of an error message in dark mode.

error\_icon\_color

error\_icon\_color\_dark

input\_background\_fill

The background of an input field.

input\_background\_fill\_dark

The background of an input field in dark mode.

input\_background\_fill\_focus

The background of an input field when focused.

input\_background\_fill\_focus\_dark

The background of an input field when focused in dark mode.

input\_background\_fill\_hover

The background of an input field when hovered over.

input\_background\_fill\_hover\_dark

The background of an input field when hovered over in dark mode.

input\_border\_color

The border color of an input field.

input\_border\_color\_dark

The border color of an input field in dark mode.

input\_border\_color\_focus

The border color of an input field when focused.

input\_border\_color\_focus\_dark

The border color of an input field when focused in dark mode.

input\_border\_color\_hover

The border color of an input field when hovered over.

input\_border\_color\_hover\_dark

The border color of an input field when hovered over in dark mode.

input\_border\_width

The border width of an input field.

input\_border\_width\_dark

The border width of an input field in dark mode.

input\_padding

The padding of an input field.

input\_placeholder\_color

The placeholder text color of an input field.

input\_placeholder\_color\_dark

The placeholder text color of an input field in dark mode.

input\_radius

The corner radius of an input field.

input\_shadow

The shadow of an input field.

input\_shadow\_dark

The shadow of an input field in dark mode.

input\_shadow\_focus

The shadow of an input field when focused.

input\_shadow\_focus\_dark

The shadow of an input field when focused in dark mode.

input\_text\_size

The text size of an input field.

input\_text\_weight

The text weight of an input field.

loader\_color

The color of the loading animation while a request is pending.

loader\_color\_dark

The color of the loading animation while a request is pending in dark mode.

slider\_color

The color of the slider in a range element.

slider\_color\_dark

The color of the slider in a range element in dark mode.

stat\_background\_fill

The background used for stats visuals (e.g. confidence bars in label).

stat\_background\_fill\_dark

The background used for stats visuals (e.g. confidence bars in label) in dark mode.

table\_border\_color

The border color of a table.

table\_border\_color\_dark

The border color of a table in dark mode.

table\_even\_background\_fill

The background of even rows in a table.

table\_even\_background\_fill\_dark

The background of even rows in a table in dark mode.

table\_odd\_background\_fill

The background of odd rows in a table.

table\_odd\_background\_fill\_dark

The background of odd rows in a table in dark mode.

table\_radius

The corner radius of a table.

table\_row\_focus

The background of a focused row in a table.

table\_row\_focus\_dark

The background of a focused row in a table in dark mode.

These set the style for buttons. You can set these to one of the dropdown values, or clear the dropdown to set a custom value.

button\_border\_width

The border width of a button.

button\_border\_width\_dark

The border width of a button in dark mode.

button\_transform\_hover

The transform animation of a button on hover.

button\_transform\_active

The transform animation of a button when pressed.

button\_transition

The transition animation duration of a button between regular, hover, and focused states.

button\_large\_padding

The padding of a button with the default "large" size.

button\_large\_radius

The corner radius of a button with the default "large" size.

button\_large\_text\_size

The text size of a button with the default "large" size.

button\_large\_text\_weight

The text weight of a button with the default "large" size.

button\_small\_padding

The padding of a button set to "small" size.

button\_small\_radius

The corner radius of a button set to "small" size.

button\_small\_text\_size

The text size of a button set to "small" size.

button\_small\_text\_weight

The text weight of a button set to "small" size.

button\_medium\_padding

The padding of a button set to "medium" size.

button\_medium\_radius

The corner radius of a button set to "medium" size.

button\_medium\_text\_size

The text size of a button set to "medium" size.

button\_medium\_text\_weight

The text weight of a button set to "medium" size.

button\_primary\_background\_fill

The background of a button of "primary" variant.

button\_primary\_background\_fill\_dark

The background of a button of "primary" variant in dark mode.

button\_primary\_background\_fill\_hover

The background of a button of "primary" variant when hovered over.

button\_primary\_background\_fill\_hover\_dark

The background of a button of "primary" variant when hovered over in dark mode.

button\_primary\_border\_color

The border color of a button of "primary" variant.

button\_primary\_border\_color\_dark

The border color of a button of "primary" variant in dark mode.

button\_primary\_border\_color\_hover

The border color of a button of "primary" variant when hovered over.

button\_primary\_border\_color\_hover\_dark

The border color of a button of "primary" variant when hovered over in dark mode.

button\_primary\_text\_color

The text color of a button of "primary" variant.

button\_primary\_text\_color\_dark

The text color of a button of "primary" variant in dark mode.

button\_primary\_text\_color\_hover

The text color of a button of "primary" variant when hovered over.

button\_primary\_text\_color\_hover\_dark

The text color of a button of "primary" variant when hovered over in dark mode.

button\_primary\_shadow

The shadow under a primary button.

button\_primary\_shadow\_hover

The shadow under a primary button when hovered over.

button\_primary\_shadow\_active

The shadow under a primary button when pressed.

button\_primary\_shadow\_dark

The shadow under a primary button in dark mode.

button\_primary\_shadow\_hover\_dark

The shadow under a primary button when hovered over in dark mode.

button\_primary\_shadow\_active\_dark

The shadow under a primary button when pressed in dark mode.

button\_secondary\_background\_fill

The background of a button of default "secondary" variant.

button\_secondary\_background\_fill\_dark

The background of a button of default "secondary" variant in dark mode.

button\_secondary\_background\_fill\_hover

The background of a button of default "secondary" variant when hovered over.

button\_secondary\_background\_fill\_hover\_dark

The background of a button of default "secondary" variant when hovered over in dark mode.

button\_secondary\_border\_color

The border color of a button of default "secondary" variant.

button\_secondary\_border\_color\_dark

The border color of a button of default "secondary" variant in dark mode.

button\_secondary\_border\_color\_hover

The border color of a button of default "secondary" variant when hovered over.

button\_secondary\_border\_color\_hover\_dark

The border color of a button of default "secondary" variant when hovered over in dark mode.

button\_secondary\_text\_color

The text color of a button of default "secondary" variant.

button\_secondary\_text\_color\_dark

The text color of a button of default "secondary" variant in dark mode.

button\_secondary\_text\_color\_hover

The text color of a button of default "secondary" variant when hovered over.

button\_secondary\_text\_color\_hover\_dark

The text color of a button of default "secondary" variant when hovered over in dark mode.

button\_secondary\_shadow

The shadow under a secondary button.

button\_secondary\_shadow\_hover

The shadow under a secondary button when hovered over.

button\_secondary\_shadow\_active

The shadow under a secondary button when pressed.

button\_secondary\_shadow\_dark

The shadow under a secondary button in dark mode.

button\_secondary\_shadow\_hover\_dark

The shadow under a secondary button when hovered over in dark mode.

button\_secondary\_shadow\_active\_dark

The shadow under a secondary button when pressed in dark mode.

button\_cancel\_background\_fill

The background of a button of "cancel" variant.

button\_cancel\_background\_fill\_dark

The background of a button of "cancel" variant in dark mode.

button\_cancel\_background\_fill\_hover

The background of a button of "cancel" variant when hovered over.

button\_cancel\_background\_fill\_hover\_dark

The background of a button of "cancel" variant when hovered over in dark mode.

button\_cancel\_border\_color

The border color of a button of "cancel" variant.

button\_cancel\_border\_color\_dark

The border color of a button of "cancel" variant in dark mode.

button\_cancel\_border\_color\_hover

The border color of a button of "cancel" variant when hovered over.

button\_cancel\_border\_color\_hover\_dark

The border color of a button of "cancel" variant when hovered over in dark mode.

button\_cancel\_text\_color

The text color of a button of "cancel" variant.

button\_cancel\_text\_color\_dark

The text color of a button of "cancel" variant in dark mode.

button\_cancel\_text\_color\_hover

The text color of a button of "cancel" variant when hovered over.

button\_cancel\_text\_color\_hover\_dark

The text color of a button of "cancel" variant when hovered over in dark mode.

Theme Builder
=============

Welcome to the theme builder. The left panel is where you create the theme. The different aspects of the theme are broken down into different tabs. Here's how to navigate them:

1.  First, set the "Source Theme". This will set the default values that you can override.
2.  Set the "Core Colors", "Core Sizing" and "Core Fonts". These are the core variables that are used to build the rest of the theme.
3.  The rest of the tabs set specific CSS theme variables. These control finer aspects of the UI. Within these theme variables, you can reference the core variables and other theme variables using the variable name preceded by an asterisk, e.g. `*primary_50` or `*body_text_color`. Clear the dropdown to set a custom value.
4.  Once you have finished your theme, click on "View Code" below to see how you can integrate the theme into your app. You can also click on "Upload to Hub" to upload your theme to the Hugging Face Hub, where others can download and use your theme.

View Code ▼

Code

[](blob:https://www.gradio.app/a09200bf-490b-4231-8138-f4e2911e7785)

›

⌄

9

1

2

3

4

5

6

import gradio as gr

  

theme \= gr.themes.Base()

  

with gr.Blocks(theme\=theme) as demo:

...

Upload to Hub ▼

You can save your theme on the Hugging Face Hub. HF API write token can be found [here](https://huggingface.co/settings/tokens).

Theme Name

Hugging Face Write Token

Version

Upload to Hub

Below this panel is a dummy app to demo your theme.

Name

Full name, including middle name. No special characters.

x 

Clear Submit

output

Slider 1

 ↺

0  100

Slider 2

 ↺

0  100

Checkbox Group

 A B  C

Panel 1
-------

Radio

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

 A B C

Dropdown

Dropdown

Option A

 Go

Image

[](https://gradio-theme-builder.hf.space/gradio_api/file=/tmp/gradio/651020e61268f3179a1645bd0793fe3ea880a865dfa6c6ffd839379e6e423570/header-image.jpg)

![](https://gradio-theme-builder.hf.space/gradio_api/file=/tmp/gradio/651020e61268f3179a1645bd0793fe3ea880a865dfa6c6ffd839379e6e423570/header-image.jpg)

Go Clear

Button 1 Upload a File Stop

Examples

Radio

Dropdown

Go

A

Option 1

Option B

true

B

Option 2

Option B, Option C

false

Dataframe

Dataframe

1

⋮

2

⋮

3

⋮

1

2

3

1

⋮

2

⋮

3

⋮

1

2

3

4

5

6

7

8

9

JSON

{

"a": 1 ,

"b": 2 ,

"c": {

"test": "a" ,

"test2": \[

"0": 1 ,

"1": 2 ,

"2": 3

\]

}

}

Label

cat
---

cat

70%

dog

20%

fish

10%

File

Drop File Here \- or - Click to Upload

ColorPicker

Video

0:00 / 0:31

[](https://gradio-theme-builder.hf.space/gradio_api/file=/tmp/gradio/e74d842f6c8d648dc4640975b5da49f7854f268d9418acee650034d9239c4351/world.mp4)

Gallery

![lion](https://gradio-static-files.s3.us-west-2.amazonaws.com/lion.jpg)

lion

![logo](https://gradio-static-files.s3.us-west-2.amazonaws.com/logo.png)

logo

![tower](https://gradio-static-files.s3.us-west-2.amazonaws.com/tower.jpg)

tower

Chatbot

.cls-1 { fill: none; }

Hello

Hi

MultimodalTextbox

Add messages

Advanced Settings ▼

Hello

Chatbot control 1 

Chatbot control 2 

Chatbot control 3 

Textbox

\[

\]

[gradio/theme\_builder](https://huggingface.co/spaces/gradio/theme_builder) built with [Gradio](https://gradio.app). Hosted on [![Hugging Face Space](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='10'%20height='10'%20fill='none'%3e%3cpath%20fill='%23FF3270'%20d='M1.93%206.03v2.04h2.04V6.03H1.93Z'/%3e%3cpath%20fill='%23861FFF'%20d='M6.03%206.03v2.04h2.04V6.03H6.03Z'/%3e%3cpath%20fill='%23097EFF'%20d='M1.93%201.93v2.04h2.04V1.93H1.93Z'/%3e%3cpath%20fill='%23000'%20fill-rule='evenodd'%20d='M.5%201.4c0-.5.4-.9.9-.9h3.1a.9.9%200%200%201%20.87.67A2.44%202.44%200%200%201%209.5%202.95c0%20.65-.25%201.24-.67%201.68.39.1.67.46.67.88v3.08c0%20.5-.4.91-.9.91H1.4a.9.9%200%200%201-.9-.9V1.4Zm1.43.53v2.04h2.04V1.93H1.93Zm0%206.14V6.03h2.04v2.04H1.93Zm4.1%200V6.03h2.04v2.04H6.03Zm0-5.12a1.02%201.02%200%201%201%202.04%200%201.02%201.02%200%200%201-2.04%200Z'%20clip-rule='evenodd'/%3e%3cpath%20fill='%23FFD702'%20d='M7.05%201.93a1.02%201.02%200%201%200%200%202.04%201.02%201.02%200%200%200%200-2.04Z'/%3e%3c/svg%3e) Spaces](https://huggingface.co/spaces)

You can use the Theme Builder running on Spaces above, though it runs much faster when you launch it locally via `gr.themes.builder()`.

As you edit the values in the Theme Builder, the app will preview updates in real time. You can download the code to generate the theme you've created so you can use it in any Gradio app.

In the rest of the guide, we will cover building themes programmatically.

Extending Themes via the Constructor[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#extending-themes-via-the-constructor)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Although each theme has hundreds of CSS variables, the values for most these variables are drawn from 8 core variables which can be set through the constructor of each prebuilt theme. Modifying these 8 arguments allows you to quickly change the look and feel of your app.

### Core Colors[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#core-colors)

The first 3 constructor arguments set the colors of the theme and are `gradio.themes.Color` objects. Internally, these Color objects hold brightness values for the palette of a single hue, ranging from 50, 100, 200..., 800, 900, 950. Other CSS variables are derived from these 3 colors.

The 3 color constructor arguments are:

*   `primary_hue`: This is the color draws attention in your theme. In the default theme, this is set to `gradio.themes.colors.orange`.
*   `secondary_hue`: This is the color that is used for secondary elements in your theme. In the default theme, this is set to `gradio.themes.colors.blue`.
*   `neutral_hue`: This is the color that is used for text and other neutral elements in your theme. In the default theme, this is set to `gradio.themes.colors.gray`.

You could modify these values using their string shortcuts, such as

    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as demo:
        ...

or you could use the `Color` objects directly, like this:

    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as demo:
        ...

Predefined colors are:

*   `slate`
*   `gray`
*   `zinc`
*   `neutral`
*   `stone`
*   `red`
*   `orange`
*   `amber`
*   `yellow`
*   `lime`
*   `green`
*   `emerald`
*   `teal`
*   `cyan`
*   `sky`
*   `blue`
*   `indigo`
*   `violet`
*   `purple`
*   `fuchsia`
*   `pink`
*   `rose`

You could also create your own custom `Color` objects and pass them in.

### Core Sizing[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#core-sizing)

The next 3 constructor arguments set the sizing of the theme and are `gradio.themes.Size` objects. Internally, these Size objects hold pixel size values that range from `xxs` to `xxl`. Other CSS variables are derived from these 3 sizes.

*   `spacing_size`: This sets the padding within and spacing between elements. In the default theme, this is set to `gradio.themes.sizes.spacing_md`.
*   `radius_size`: This sets the roundedness of corners of elements. In the default theme, this is set to `gradio.themes.sizes.radius_md`.
*   `text_size`: This sets the font size of text. In the default theme, this is set to `gradio.themes.sizes.text_md`.

You could modify these values using their string shortcuts, such as

    with gr.Blocks(theme=gr.themes.Default(spacing_size="sm", radius_size="none")) as demo:
        ...

or you could use the `Size` objects directly, like this:

    with gr.Blocks(theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_none)) as demo:
        ...

The predefined size objects are:

*   `radius_none`
*   `radius_sm`
*   `radius_md`
*   `radius_lg`
*   `spacing_sm`
*   `spacing_md`
*   `spacing_lg`
*   `text_sm`
*   `text_md`
*   `text_lg`

You could also create your own custom `Size` objects and pass them in.

### Core Fonts[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#core-fonts)

The final 2 constructor arguments set the fonts of the theme. You can pass a list of fonts to each of these arguments to specify fallbacks. If you provide a string, it will be loaded as a system font. If you provide a `gradio.themes.GoogleFont`, the font will be loaded from Google Fonts.

*   `font`: This sets the primary font of the theme. In the default theme, this is set to `gradio.themes.GoogleFont("IBM Plex Sans")`.
*   `font_mono`: This sets the monospace font of the theme. In the default theme, this is set to `gradio.themes.GoogleFont("IBM Plex Mono")`.

You could modify these values such as the following:

    with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])) as demo:
        ...

Extending Themes via `.set()`[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#extending-themes-via-set)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

You can also modify the values of CSS variables after the theme has been loaded. To do so, use the `.set()` method of the theme object to get access to the CSS variables. For example:

    theme = gr.themes.Default(primary_hue="blue").set(
        loader_color="#FF0000",
        slider_color="#FF0000",
    )
    
    with gr.Blocks(theme=theme) as demo:
        ...

In the example above, we've set the `loader_color` and `slider_color` variables to `#FF0000`, despite the overall `primary_color` using the blue color palette. You can set any CSS variable that is defined in the theme in this manner.

Your IDE type hinting should help you navigate these variables. Since there are so many CSS variables, let's take a look at how these variables are named and organized.

### CSS Variable Naming Conventions[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#css-variable-naming-conventions)

CSS variable names can get quite long, like `button_primary_background_fill_hover_dark`! However they follow a common naming convention that makes it easy to understand what they do and to find the variable you're looking for. Separated by underscores, the variable name is made up of:

1.  The target element, such as `button`, `slider`, or `block`.
2.  The target element type or sub-element, such as `button_primary`, or `block_label`.
3.  The property, such as `button_primary_background_fill`, or `block_label_border_width`.
4.  Any relevant state, such as `button_primary_background_fill_hover`.
5.  If the value is different in dark mode, the suffix `_dark`. For example, `input_border_color_focus_dark`.

Of course, many CSS variable names are shorter than this, such as `table_border_color`, or `input_shadow`.

### CSS Variable Organization[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#css-variable-organization)

Though there are hundreds of CSS variables, they do not all have to have individual values. They draw their values by referencing a set of core variables and referencing each other. This allows us to only have to modify a few variables to change the look and feel of the entire theme, while also getting finer control of individual elements that we may want to modify.

#### Referencing Core Variables[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#referencing-core-variables)

To reference one of the core constructor variables, precede the variable name with an asterisk. To reference a core color, use the `*primary_`, `*secondary_`, or `*neutral_` prefix, followed by the brightness value. For example:

    theme = gr.themes.Default(primary_hue="blue").set(
        button_primary_background_fill="*primary_200",
        button_primary_background_fill_hover="*primary_300",
    )

In the example above, we've set the `button_primary_background_fill` and `button_primary_background_fill_hover` variables to `*primary_200` and `*primary_300`. These variables will be set to the 200 and 300 brightness values of the blue primary color palette, respectively.

Similarly, to reference a core size, use the `*spacing_`, `*radius_`, or `*text_` prefix, followed by the size value. For example:

    theme = gr.themes.Default(radius_size="md").set(
        button_primary_border_radius="*radius_xl",
    )

In the example above, we've set the `button_primary_border_radius` variable to `*radius_xl`. This variable will be set to the `xl` setting of the medium radius size range.

#### Referencing Other Variables[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#referencing-other-variables)

Variables can also reference each other. For example, look at the example below:

    theme = gr.themes.Default().set(
        button_primary_background_fill="#FF0000",
        button_primary_background_fill_hover="#FF0000",
        button_primary_border="#FF0000",
    )

Having to set these values to a common color is a bit tedious. Instead, we can reference the `button_primary_background_fill` variable in the `button_primary_background_fill_hover` and `button_primary_border` variables, using a `*` prefix.

    theme = gr.themes.Default().set(
        button_primary_background_fill="#FF0000",
        button_primary_background_fill_hover="*button_primary_background_fill",
        button_primary_border="*button_primary_background_fill",
    )

Now, if we change the `button_primary_background_fill` variable, the `button_primary_background_fill_hover` and `button_primary_border` variables will automatically update as well.

This is particularly useful if you intend to share your theme - it makes it easy to modify the theme without having to change every variable.

Note that dark mode variables automatically reference each other. For example:

    theme = gr.themes.Default().set(
        button_primary_background_fill="#FF0000",
        button_primary_background_fill_dark="#AAAAAA",
        button_primary_border="*button_primary_background_fill",
        button_primary_border_dark="*button_primary_background_fill_dark",
    )

`button_primary_border_dark` will draw its value from `button_primary_background_fill_dark`, because dark mode always draw from the dark version of the variable.

Creating a Full Theme[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#creating-a-full-theme)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Let's say you want to create a theme from scratch! We'll go through it step by step - you can also see the source of prebuilt themes in the gradio source repo for reference - [here's the source](https://github.com/gradio-app/gradio/blob/main/gradio/themes/monochrome.py) for the Monochrome theme.

Our new theme class will inherit from `gradio.themes.Base`, a theme that sets a lot of convenient defaults. Let's make a simple demo that creates a dummy theme called Seafoam, and make a simple app that uses it.

    import gradio as gr
    from gradio.themes.base import Base
    import time
    
    class Seafoam(Base):
        pass
    
    seafoam = Seafoam()
    
    with gr.Blocks(theme=seafoam) as demo:
        textbox = gr.Textbox(label="Name")
        slider = gr.Slider(label="Count", minimum=0, maximum=100, step=1)
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")
        output = gr.Textbox(label="Output")
    
        def repeat(name, count):
            time.sleep(3)
            return name * count
    
        button.click(repeat, [textbox, slider], output)
    
    demo.launch()
    

The Base theme is very barebones, and uses `gr.themes.Blue` as it primary color - you'll note the primary button and the loading animation are both blue as a result. Let's change the defaults core arguments of our app. We'll overwrite the constructor and pass new defaults for the core constructor arguments.

We'll use `gr.themes.Emerald` as our primary color, and set secondary and neutral hues to `gr.themes.Blue`. We'll make our text larger using `text_lg`. We'll use `Quicksand` as our default font, loaded from Google Fonts.

    from __future__ import annotations
    from typing import Iterable
    import gradio as gr
    from gradio.themes.base import Base
    from gradio.themes.utils import colors, fonts, sizes
    import time
    
    class Seafoam(Base):
        def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.emerald,
            secondary_hue: colors.Color | str = colors.blue,
            neutral_hue: colors.Color | str = colors.gray,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_lg,
            font: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("Quicksand"),
                "ui-sans-serif",
                "sans-serif",
            ),
            font_mono: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        ):
            super().__init__(
                primary_hue=primary_hue,
                secondary_hue=secondary_hue,
                neutral_hue=neutral_hue,
                spacing_size=spacing_size,
                radius_size=radius_size,
                text_size=text_size,
                font=font,
                font_mono=font_mono,
            )
    
    seafoam = Seafoam()
    
    with gr.Blocks(theme=seafoam) as demo:
        textbox = gr.Textbox(label="Name")
        slider = gr.Slider(label="Count", minimum=0, maximum=100, step=1)
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")
        output = gr.Textbox(label="Output")
    
        def repeat(name, count):
            time.sleep(3)
            return name * count
    
        button.click(repeat, [textbox, slider], output)
    
    demo.launch()
    

See how the primary button and the loading animation are now green? These CSS variables are tied to the `primary_hue` variable.

Let's modify the theme a bit more directly. We'll call the `set()` method to overwrite CSS variable values explicitly. We can use any CSS logic, and reference our core constructor arguments using the `*` prefix.

    from __future__ import annotations
    from typing import Iterable
    import gradio as gr
    from gradio.themes.base import Base
    from gradio.themes.utils import colors, fonts, sizes
    import time
    
    class Seafoam(Base):
        def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.emerald,
            secondary_hue: colors.Color | str = colors.blue,
            neutral_hue: colors.Color | str = colors.blue,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_lg,
            font: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("Quicksand"),
                "ui-sans-serif",
                "sans-serif",
            ),
            font_mono: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        ):
            super().__init__(
                primary_hue=primary_hue,
                secondary_hue=secondary_hue,
                neutral_hue=neutral_hue,
                spacing_size=spacing_size,
                radius_size=radius_size,
                text_size=text_size,
                font=font,
                font_mono=font_mono,
            )
            super().set(
                body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
                body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
                button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
                button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
                button_primary_text_color="white",
                button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
                slider_color="*secondary_300",
                slider_color_dark="*secondary_600",
                block_title_text_weight="600",
                block_border_width="3px",
                block_shadow="*shadow_drop_lg",
                button_primary_shadow="*shadow_drop_lg",
                button_large_padding="32px",
            )
    
    seafoam = Seafoam()
    
    with gr.Blocks(theme=seafoam) as demo:
        textbox = gr.Textbox(label="Name")
        slider = gr.Slider(label="Count", minimum=0, maximum=100, step=1)
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")
        output = gr.Textbox(label="Output")
    
        def repeat(name, count):
            time.sleep(3)
            return name * count
    
        button.click(repeat, [textbox, slider], output)
    
    demo.launch()
    

Look how fun our theme looks now! With just a few variable changes, our theme looks completely different.

You may find it helpful to explore the [source code of the other prebuilt themes](https://github.com/gradio-app/gradio/blob/main/gradio/themes) to see how they modified the base theme. You can also find your browser's Inspector useful to select elements from the UI and see what CSS variables are being used in the styles panel.

Sharing Themes[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#sharing-themes)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Once you have created a theme, you can upload it to the HuggingFace Hub to let others view it, use it, and build off of it!

### Uploading a Theme[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#uploading-a-theme)

There are two ways to upload a theme, via the theme class instance or the command line. We will cover both of them with the previously created `seafoam` theme.

*   Via the class instance

Each theme instance has a method called `push_to_hub` we can use to upload a theme to the HuggingFace hub.

    seafoam.push_to_hub(repo_name="seafoam",
                        version="0.0.1",
    					hf_token="<token>")

*   Via the command line

First save the theme to disk

    seafoam.dump(filename="seafoam.json")

Then use the `upload_theme` command:

    upload_theme\
    "seafoam.json"\
    "seafoam"\
    --version "0.0.1"\
    --hf_token "<token>"

In order to upload a theme, you must have a HuggingFace account and pass your [Access Token](https://huggingface.co/docs/huggingface_hub/quick-start#login) as the `hf_token` argument. However, if you log in via the [HuggingFace command line](https://huggingface.co/docs/huggingface_hub/quick-start#login) (which comes installed with `gradio`), you can omit the `hf_token` argument.

The `version` argument lets you specify a valid [semantic version](https://www.geeksforgeeks.org/introduction-semantic-versioning/) string for your theme. That way your users are able to specify which version of your theme they want to use in their apps. This also lets you publish updates to your theme without worrying about changing how previously created apps look. The `version` argument is optional. If omitted, the next patch version is automatically applied.

### Theme Previews[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#theme-previews)

By calling `push_to_hub` or `upload_theme`, the theme assets will be stored in a [HuggingFace space](https://huggingface.co/docs/hub/spaces-overview).

The theme preview for our seafoam theme is here: [seafoam preview](https://huggingface.co/spaces/gradio/seafoam).

### Discovering Themes[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#discovering-themes)

The [Theme Gallery](https://huggingface.co/spaces/gradio/theme-gallery) shows all the public gradio themes. After publishing your theme, it will automatically show up in the theme gallery after a couple of minutes.

You can sort the themes by the number of likes on the space and from most to least recently created as well as toggling themes between light and dark mode.

### Downloading[![](data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20fill='%23808080'%20viewBox='0%200%20640%20512'%3e%3c!--!%20Font%20Awesome%20Pro%206.0.0%20by%20@fontawesome%20-%20https://fontawesome.com%20License%20-%20https://fontawesome.com/license%20(Commercial%20License)%20Copyright%202022%20Fonticons,%20Inc.%20--%3e%3cpath%20d='M172.5%20131.1C228.1%2075.51%20320.5%2075.51%20376.1%20131.1C426.1%20181.1%20433.5%20260.8%20392.4%20318.3L391.3%20319.9C381%20334.2%20361%20337.6%20346.7%20327.3C332.3%20317%20328.9%20297%20339.2%20282.7L340.3%20281.1C363.2%20249%20359.6%20205.1%20331.7%20177.2C300.3%20145.8%20249.2%20145.8%20217.7%20177.2L105.5%20289.5C73.99%20320.1%2073.99%20372%20105.5%20403.5C133.3%20431.4%20177.3%20435%20209.3%20412.1L210.9%20410.1C225.3%20400.7%20245.3%20404%20255.5%20418.4C265.8%20432.8%20262.5%20452.8%20248.1%20463.1L246.5%20464.2C188.1%20505.3%20110.2%20498.7%2060.21%20448.8C3.741%20392.3%203.741%20300.7%2060.21%20244.3L172.5%20131.1zM467.5%20380C411%20436.5%20319.5%20436.5%20263%20380C213%20330%20206.5%20251.2%20247.6%20193.7L248.7%20192.1C258.1%20177.8%20278.1%20174.4%20293.3%20184.7C307.7%20194.1%20311.1%20214.1%20300.8%20229.3L299.7%20230.9C276.8%20262.1%20280.4%20306.9%20308.3%20334.8C339.7%20366.2%20390.8%20366.2%20422.3%20334.8L534.5%20222.5C566%20191%20566%20139.1%20534.5%20108.5C506.7%2080.63%20462.7%2076.99%20430.7%2099.9L429.1%20101C414.7%20111.3%20394.7%20107.1%20384.5%2093.58C374.2%2079.2%20377.5%2059.21%20391.9%2048.94L393.5%2047.82C451%206.731%20529.8%2013.25%20579.8%2063.24C636.3%20119.7%20636.3%20211.3%20579.8%20267.7L467.5%20380z'/%3e%3c/svg%3e)](#downloading)

To use a theme from the hub, use the `from_hub` method on the `ThemeClass` and pass it to your app:

    my_theme = gr.Theme.from_hub("gradio/seafoam")
    
    with gr.Blocks(theme=my_theme) as demo:
        ....

You can also pass the theme string directly to `Blocks` or `Interface` (`gr.Blocks(theme="gradio/seafoam")`)

You can pin your app to an upstream theme version by using semantic versioning expressions.

For example, the following would ensure the theme we load from the `seafoam` repo was between versions `0.0.1` and `0.1.0`:

    with gr.Blocks(theme="gradio/seafoam@>=0.0.1,<0.1.0") as demo:
        ....

Enjoy creating your own themes! If you make one you're proud of, please share it with the world by uploading it to the hub! If you tag us on [Twitter](https://twitter.com/gradio) we can give your theme a shout out!

[

←

Styling The Gradio Dataframe



](../guides/styling-the-gradio-dataframe/)[

Understanding Gradio Share Links

→

](../guides/understanding-gradio-share-links/)