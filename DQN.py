from config_DQN.trainier import DQN_Trainer


if __name__ == '__main__':
    trainer = DQN_Trainer('./config/01_DQN_Breakout_config.json')
    trainer.run()