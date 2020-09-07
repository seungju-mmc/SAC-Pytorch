from config_SAC.trainer import SAC_Trainer


if __name__ == '__main__':
    trainer = SAC_Trainer('./config/04_SAC_HN_config.json')
    trainer.run()