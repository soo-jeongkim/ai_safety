def play(env = get_new_env(), path = None):
    time_step = env.reset()
    observation = time_step.observation["board"]

    while True:
        if path is None: print_game(convert_board_num2str(observation))

        if time_step.step_type == environment.StepType.LAST:
            if path is None: print("GAME OVER")
            break

        print(time_step)

        if path is None:
            user_input = input("Next move>")
            clear_output()

        else:
            user_input = "d" if len(path) == 0 else path.pop(0) 

        # validate input
        if user_input not in valid_action_strings:
            print("Try an action in", valid_action_strings.keys())
            continue
        elif user_input == "q":
            break

        # env takes an integer action - convert input
        action = valid_action_strings[user_input]

        # take the action in the environment
        time_step = env.step(action)
        observation = time_step.observation["board"]

    return 0.0 ## TODO Exercise 1: compute the total reward over this episode

#play()