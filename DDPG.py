from config_DDPG.trainer import DDPG_Trainer


if __name__ == '__main__':
    trainer = DDPG_Trainer('./config/03_DDPG_Hopper_config.json')
    trainer.run()