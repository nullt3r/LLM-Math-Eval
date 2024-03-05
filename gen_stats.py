from openai import OpenAI
import re
import numpy as np
import matplotlib.pyplot as plt

# Configuration for the OpenAI client
client = OpenAI(base_url="http://192.168.1.223:1234/v1", api_key="not-needed")


def generate_questions(operation, num_questions, max_value=100000):
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


def ask_question(question):
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are a Super Computer Math calculator. Return only the result with three decimal places maximum. No other text is allowed."},
            {"role": "user", "content": question},
        ],
        temperature=1.2,
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
operation_accuracies = {op: [] for op in operations}
low_accuracy_threshold = 0.8
low_accuracy_counts = {op: 0 for op in operations}

for operation in operations:
    questions = generate_questions(operation, num_questions_per_operation)
    for question, num1, num2 in questions:
        correct_answer = calculate_correct_answer(operation, num1, num2)
        model_answer = ask_question(question)
        accuracy = calculate_accuracy(correct_answer, model_answer)
        operation_accuracies[operation].append(accuracy)
        if accuracy < low_accuracy_threshold:
            low_accuracy_counts[operation] += 1

        # Logging the results
        print(f"Question: {question} | Expected: {correct_answer:.3f}, Model: {model_answer:.3f} | Accuracy: {accuracy * 100:.2f}%")

# Rest of the plotting code remains unchanged

# Calculating average accuracy and low accuracy percentage
avg_accuracies = {op: np.mean(accuracies) for op, accuracies in operation_accuracies.items()}
low_accuracy_percentage = {op: count / num_questions_per_operation for op, count in low_accuracy_counts.items()}

# Plotting the average accuracy per operation with low accuracy highlighted
bar_width = 0.4
indices = np.arange(len(avg_accuracies))

plt.bar(indices, list(avg_accuracies.values()), bar_width, align='center', alpha=0.7, color="grey", label='Average Accuracy')
plt.bar(indices, list(low_accuracy_percentage.values()), bar_width, align='center', alpha=0.7, color='red', label='Low Accuracy (<80%)')

plt.xticks(indices, list(avg_accuracies.keys()))
plt.xlabel('Operation')
plt.ylabel('Accuracy (0.0-1.0)')
plt.title('Average Accuracy per Operation with Low Accuracy Highlight')

# Adding average accuracy as text on bars
for i, (op, accuracy) in enumerate(avg_accuracies.items()):
    plt.text(i, accuracy, f"{accuracy * 100:.0f}%", ha='center', va='bottom')

# Adding low accuracy percentage as text on red bars
for i, (op, low_acc) in enumerate(low_accuracy_percentage.items()):
    if low_acc > 0:  # Only display if there's a notable low accuracy
        plt.text(i, low_acc, f"{low_acc * 100:.0f}%", ha='center', va='bottom', color='black')

plt.legend()
plt.show()
