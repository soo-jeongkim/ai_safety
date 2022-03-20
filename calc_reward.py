def play(env = get_new_env(), path = None):
  time_step = env.reset()

  if path is None: 
    print("Always choose an input from the list:")
    print(valid_action_strings.keys())
    print("Start state")
    print(time_step)

  observation = time_step.observation["board"]

  tot = 0

  while True:
    if path is None: print_game(convert_board_num2str(observation))

    if time_step.reward is not None:
      tot += time_step.reward

    if time_step.step_type == environment.StepType.LAST:
      if path is None: print("Game over")
      break
      
    if path is None:
      user_input = input("Next move>")
      clear_output()
      
    else:
      user_input = "d" if len(path) == 0 else path.pop(0)

    #validate input
    if user_input not in valid_action_strings:
      print("Try an action in:", valid_action_strings.key())
      continue
    elif user_input == "q":
      break
    
    #env takes an integer action - convert input
    action = valid_action_strings[user_input]

    # take the action in the environment
    time_step = env.step(action)
    observation = time_step.observation["board"]
  
  return tot

# play()