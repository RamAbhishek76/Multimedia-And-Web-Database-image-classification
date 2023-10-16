print("\n1. Task 7\n2. Task 8\n3. Task 9\n4. Task 10\n5. Task 11\n")
task = int(input(("Select one of the tasks from above: ")))
tasks = ['phase_2_task_7.py', 'phase_2_task_8.py', 'phase_2_task_9.py', 'phase_2_task_10.py', 'phase_2_task_11.py']

if task in range(1, 6):
    task_file = tasks[task-1]
    try:
        module = __import__(task_file[:-3])
        module.main()
    except ImportError:
        print(f"Error: {task_file} not found.")
else:
    print("Invalid task number. Please select a number between 1 and 5.")
