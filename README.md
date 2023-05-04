To do/test :
-[ ] Use pretrained weight for wideresnet ?
-[ ] dans net/wideresent remplacer self.classifier = nn.Linear(channels[3], num_classes)
par un MLP : idée est de mettre moins de pression sur
- [ ] ajouter le fait qu'on puisse choisir un facteur pour 
prendre moins en compte la partie non supervisée de la loss supcon

- [ ] question : la partie x_ulb_w est detachée, pour calculer la partie ce_unsuper 
(qui entraine la projection lineaure), il faudrait en fait attacher le gradient )
D; ailleur si c'est vriament détahé le fait de rajouter la loss ce_unsupervised ne devrait rien changer

Comment ajouter la tete de projection contrastive de maniere clean ?

- [ ] multi crop ?
- [ ] cutmix ?