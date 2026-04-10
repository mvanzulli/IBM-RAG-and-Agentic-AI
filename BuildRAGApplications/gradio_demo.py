import gradio as gr

def add_numbers(Num1, Num2):
    return Num1 + Num2

# Define the interface
demo = gr.Interface(
    fn=add_numbers, 
    inputs=[gr.Number(), gr.Number()], # Create two numerical input fields where users can enter numbers
    outputs=gr.Number() # Create numerical output fields
)

# Launch the interface
demo.launch(server_name="127.0.0.1", server_port= 7860)

"""
Can you create a Gradio application that can combine two input sentences together?
 Use what you know from the demo and your Python knowledge to create this app.
 Take your time to complete this exercise.
"""

def combine(a, b): return a + " " + b

demo = gr.Interface(
    fn=combine, 
    inputs=[gr.Textbox(label = "input 1"), gr.Textbox(label = "input 2")], 
    outputs=gr.Textbox(label="Output text") 
)

# Launch the interface
demo.launch(server_name="127.0.0.1", server_port= 7860)

