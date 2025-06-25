bro allora:
-reproducible_ac_exp_1: training con la vecchia critic (algoritmo actor-critic, non è una vera loss)
-reproducible_ac_exp_2: training con la vecchia critic + entropy loss
-reproducible_ac_exp_3: training con la vecchia critic + rewards forzati per la consegna 
-reproducible_ac_exp_4: training con la vecchia critic + entropy loss + rewards forzati per la consegna

i risultati sono soddisfacenti anche se teoricamente come viene trainata la critic non ha nessun senso logico....

-new_loss_critic_ac_exp_1: training con la nuova loss + rewards forzati per la consegna. 
-new_loss_critic_ac_exp_2: training con la nuova loss + rewards forzati per la consegna + learning rate più elevati (1e-5 entrambi).
-new_loss_critic_ppo_exp_1: ppo, training con la nuova loss + rewards forzati per la consegna 
-new_loss_critic_ppo_exp_2: ppo, training con la nuova loss + rewards forzati per la consegna + learning rate elevati (1e-4, 5e-5)
-new_loss_critic_ppo_exp_3: ppo, training con la nuova loss + learning rate elevati (1e-4, 5e-5)
-new_loss_critic_ppo_exp_4: ppo, training con la nuova loss + learning rate elevati (1e-4, 5e-5) + entropy loss