from dataset import *
from model import *
from config import *

if __name__ == '__main__':

  # Perfom MTCNN facial detection
  aligned = []
  names = []

  for x, y in loader:
      x_aligned, prob = mtcnn(x, return_prob=True)
      if x_aligned is not None:
          print('Face detected with probability: {:8f}'.format(prob))
          aligned.append(x_aligned)
          names.append(dataset.idx_to_class[y])

  # Calculate image embeddings
  aligned = torch.stack(aligned).to(device)
  embeddings = resnet(aligned).detach().cpu()

  # Print distance matrix for classes
  dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
  pd.DataFrame(dists, columns=names, index=names)
  

