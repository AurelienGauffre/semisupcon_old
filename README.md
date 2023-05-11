To do/test :
-[ ] Use pretrained weight for wideresnet ?
- [ ] ajouter le fait qu'on puisse choisir un facteur pour 
prendre moins en compte la partie non supervisée de la loss supcon
- Loss supcon unifiée
- Pour la CE pseudolabel, rajouter un deuxime threshold plus haut ! Et regarder le % de pseudo labelisé

- [ ] question : la partie x_ulb_w est detachée, pour calculer la partie ce_unsuper 
(qui entraine la projection lineaure), il faudrait en fait attacher le gradient )
D; ailleur si c'est vriament détahé le fait de rajouter la loss ce_unsupervised ne devrait rien changer

Comment ajouter la tete de projection contrastive de maniere clean ?

- [ ] multi crop ?
- [ ] cutmix ?