import ipywidgets as widgets
from IPython.display import HTML, clear_output, display


def change_topic_names(topic_list, id_list):
    new_names = []

    # Create a list of Text widgets for entering new names with IDs as descriptions
    text_widgets = []

    for i, (topic, topic_id) in enumerate(zip(topic_list, id_list)):
        text_widget = widgets.Text(value=topic, description=f"{topic_id}:")
        original_topic_label = widgets.Label(value=topic)
        text_widgets.append(widgets.HBox([text_widget, original_topic_label]))

    # Create a title widget
    title_widget = widgets.HTML("Manually input the new topic names: ")

    # Combine the title, Text widgets, and a button in a VBox
    container = widgets.VBox([title_widget] + text_widgets)

    # Create an Output widget for displaying the applied message
    output = widgets.Output()

    def apply_changes(button):
        new_names.clear()
        for i, text_widget in enumerate(text_widgets):
            new_name = text_widget.children[0].value.strip()
            if new_name == "":
                new_names.append(text_widget.children[1].value)  # Keep the same name
            else:
                new_names.append(new_name)

        # Display the applied message
        with output:
            clear_output()
            display(
                HTML(
                    '<span style="color: green; font-weight: bold;">Changes Applied!</span>'
                )
            )

    # Create a button to apply changes with text color #2596be and bold description
    apply_button = widgets.Button(
        description="Apply Changes",
        style={"button_color": "#2596be", "color": "#2596be"},
    )

    # Attach the apply_changes function to the button's on_click event
    apply_button.on_click(apply_changes)

    # Display the container, apply button, and output widget
    display(container, apply_button, output)

    return new_names
