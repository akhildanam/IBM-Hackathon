# IBM-Hackathon
Analytics Vidya Hackathon

Building recommendation engine for online retail portals using Machine learning techniques

Libraries used -

Pandas(0.25.0)

Numpy(1.16.4)

Matplotlib

Scikit-learn(0.21.2)
keras


Model Used - RNN

4 Layered LSTM Network

Drop-out used in U, V (rate = 0.2)

Posed as Multi-label Classification problem instead of Recommender system

Sigmoid Activation layer & Binary Cross Entropy Loss

Pre-padded the input time sequences with maximum time step = 32

Training & Testing:

Experiment 1: K-Hot Vectors as Transactions as Input (32 time steps)

Experiment 2: Embedding Layer + 4 Layer LSTM with product sequence as Transactions

Experiment 3: Pre-trained Embeddings (Averaged over a trip)

Leaderboard Rank - 8/3500
