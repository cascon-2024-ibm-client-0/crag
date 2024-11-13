def saveGraph(fileName: str, graph):
    # Assuming app.get_graph().draw_mermaid_png() returns image data in bytes
    image_data = graph.get_graph().draw_mermaid_png()

    # Specify the file path where you want to save the image
    file_path = f"{fileName}.png"

    # Write the image data to the file
    with open(file_path, 'wb') as file:
        file.write(image_data)