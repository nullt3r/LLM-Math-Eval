from openai import OpenAI
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# Configuration for the OpenAI client
client = OpenAI(base_url="http://192.168.1.223:1234/v1", api_key="not-needed")


def generate_questions(operation, num_questions, max_value=1000000):
    questions = []
    for _ in range(num_questions):
        # Generate two random numbers based on the operation
        if operation in ['+', '-']:
            num1 = np.random.randint(0, max_value + 1)
            num2 = np.random.randint(0, max_value + 1)
        elif operation == '*':
            # Limit the range for multiplication
            sqrt_max = int(max_value ** 0.5)
            num1 = np.random.randint(1, sqrt_max + 1)
            num2 = np.random.randint(1, min(max_value // num1, sqrt_max) + 1)
        elif operation == '/':
            # Ensure divisor and dividend are not the same
            num2 = np.random.randint(1, max_value + 1)
            num1 = num2 * np.random.randint(1, max_value // num2 + 1)
            while num1 == num2:
                num2 = np.random.randint(1, max_value + 1)
                num1 = num2 * np.random.randint(1, max_value // num2 + 1)

        question = f"What is {num1} {operation} {num2}?"
        questions.append((question, num1, num2))
    return questions


def ask_question(question, temperature):
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are a Super Computer Math calculator. Return only the result with three decimal places maximum. No other text is allowed."},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=12
    )
    answer = completion.choices[0].message.content.strip()
    match = re.search(r'-?\d+\.?\d*', answer)
    return float(match.group()) if match else None


def calculate_accuracy(correct_answer, model_answer):
    model_answer = float(model_answer) if model_answer is not None else None
    if model_answer is None:
        return 0.0
    try:
        return max(0.0, 1.0 - abs(model_answer - correct_answer) / abs(correct_answer))
    except ZeroDivisionError:
        return 1.0 if model_answer == correct_answer else 0.0


def calculate_correct_answer(operation, num1, num2):
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        return num1 / num2
    else:
        raise ValueError("Invalid operation")


operations = ['+', '-', '*', '/']
num_questions_per_operation = 50
temperature_steps = np.arange(0.0, 2.2, 0.2) # From 0.6 to 2.0 in steps of 0.2
temperature_results = {temp: {op: [] for op in operations} for temp in temperature_steps}

for temperature in temperature_steps:
    operation_accuracies = {op: [] for op in operations}
    low_accuracy_counts = {op: 0 for op in operations}

    for operation in operations:
        questions = generate_questions(operation, num_questions_per_operation)
        for question, num1, num2 in questions:
            correct_answer = calculate_correct_answer(operation, num1, num2)
            model_answer = ask_question(question, temperature)
            accuracy = calculate_accuracy(correct_answer, model_answer)
            operation_accuracies[operation].append(accuracy)
            if accuracy < 0.8:
                low_accuracy_counts[operation] += 1
            print(f"Question: {question} | Expected: {correct_answer:.3f}, Model: {model_answer:.3f} | Temperature: {temperature:.3f} | Accuracy: {accuracy * 100:.2f}%")

    for op in operations:
        avg_acc = np.mean(operation_accuracies[op])
        low_acc_percentage = low_accuracy_counts[op] / num_questions_per_operation
        temperature_results[temperature][op] = (avg_acc, low_acc_percentage)

# Plotting the results for each operation across temperatures
style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Grid of 2x2 for four operations

for i, operation in enumerate(operations):
    # Convert proportions to percentages
    avg_accuracies = [temperature_results[temp][operation][0] * 100 for temp in temperature_steps]
    low_accuracies = [temperature_results[temp][operation][1] * 100 for temp in temperature_steps]

    ax = axs[i//2, i%2]
    ax.plot(temperature_steps, avg_accuracies, label='Average Accuracy', marker='o', color="#7509E6")
    ax.plot(temperature_steps, low_accuracies, label='Low Accuracy (<80%)', marker='x', color='#F44336')
    ax.set_title(f'Operation: {operation}')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Accuracy (Percentage)')
    ax.legend()
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()
