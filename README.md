Si save_name est sur None, alors le nom est automatiquenment généré a {algo}_{dataset}_{model}_{date}_{time} sinon on choisis
To do/test :
le wandb name est juste ajouté au début )c'est un prefix)

- [x] Use pretrained weight for wideresnet A VOIR JE ME SOUVIENS PLUS
- [ ] checks that these pretrained works are loaded correctly : why the loss is not continous after loader
- [ ] weights
- [ ] essayer avec préentrainement simclr : fixmatch vs notre method
- [ ] ajouter


- [ ] adaptation to simsiam : for the pretraining could work and what about supcon loss ? Does it becomes exactly consistency reg ?
- [ ] multi crop ?
- [ ] cutmix ?

- essayer plus petie BS
- transfer learning apres preentrainement contrastif
- dropout ?