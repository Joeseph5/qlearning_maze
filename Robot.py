import random
import math

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze 
        self.valid_actions = self.maze.valid_actions # 所有有效的行动 
        self.state = None 
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        # 输出机器人的当前位置
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            epsilon = 0
        else:
            # TODO 2. Update parameters when learning
            # 更新时间
            if self.epsilon < 0.01:
                self.epsilon = 0
            else:
                # 随着时间变化更新epsilon，100:1.
                # self.epsilon -= self.t * 0.1;
                self.epsilon = self.epsilon0 / (self.t / 100 + 1) 
                # exp_value = math.exp(6 - 0.2 * self.t)
                # self.epsilon = self.epsilon0 * (exp_value / (1 + exp_value))
            self.t += 1
        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        # 判断state是否在字典中
        if state not in self.Qtable:
            # 初始化state对应键值
            self.Qtable[state] = {'u':0.0, 'r':0.0, 'd':0.0, 'l':0.0}

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            # 随机数小于epsilon开启学习
            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return random.choice(self.valid_actions)
            else:
                # TODO 7. Return action with highest q value
                return max(self.Qtable[self.state], key = self.Qtable[self.state].get)
        elif self.testing:
            # TODO 7. choose action with highest q value
            return max(self.Qtable[self.state], key = self.Qtable[self.state].get)
        else:
            # TODO 6. Return random choose aciton
            return random.choice(self.valid_actions)
    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            qOld = self.Qtable[self.state][action]
            qNextMax = float(max(self.Qtable[next_state].values()))
            self.Qtable[self.state][action] = (1 - self.alpha) * qOld + self.alpha * (r + self.gamma * qNextMax) 


    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
