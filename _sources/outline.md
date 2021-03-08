# Outline

## Figure Panels
- Show the data
    - What do the neurons look like? Plot the neurons in space, maybe show a few examples, etc.
    - What does the graph look like? Can plot adjacencies as well as some kind of graph layout possibly.
    - `[x]` Some simple descripive statistics (# nodes, # edges, # synapses, degrees, weights, etc.)
        - Table: number of nodes, number of edges, number of synapses
        - Panel of edge weight distribution
        - Panel of in vs out degree with marginals
    - (Maybe) something describing the different edge types
- A priori SBM-family model comparison and selection
    - We have some A priori information: 
        - left/right hemisphere
        - ipsilateral, contralateral, bilateral
        - loose cell types
            - sensory/motor/interneuron
            - other rough categories
    - We can fit a priori SBMs using all of the above. Some of them can even be nested (e.g. by left/right and then by ipsi/contra/bi)
    - For each, calculate the number of free parameters.
    - Do some kind of model selection/comparison and try to select the best one
    - Left/right hemisphere blockmodel (_question: how does this relate to the above, even within a model class of 2 block SBMs there are a ton of different ways to parameterize_)
        - maybe framed less as a test, more as part of the model selection question
        - testing for homophillic/assortative, the different SBM block probability hypotheses
        - Could do the above with the different 4 color graphs as well
    - have lots of tools to evaluate.
        - with the DC stuff we don't quite know how to do it. with SBM jovo thinks we do know how.
    - point of this figure is lets evaluate how good our models are
- A posteriori modeling
    - point of this one is what can we learn about the structure of the data that we dont have a priori
    - Hierarchical SBM estimation
        - should we revisit how this is estimated?
        - we never tried initializing with a prior
        - could use the best a priori as init for a posteriori
    - Leiden hierarchical SBM estimation, how are these different/not different (do we want this?)
        - this is just another way of fitting an SBM, with different constraints.
    - Comparison of model complexity (dDCSBM, SBM, RDPG-d, etc.)
    - Follow up with brain images labeling the neurons
    - Want to justify by looking at the neurobiology 
- Ok, so maybe we should include vertex attributes
    - Embedding with node covariates for example
    - Maybe we just show?
    - Figure out how to evaluate?
- (maybe) multigraph model selection
- Graph matching methods figure (I don't think these results from Youngser/Carey ever went to a paper anywhere? so I assume CEP would be okay with them here? And we should be able to replicate/improve in python now.)
    - Show some examples of pairs in space.
    - Show results of vanilla GM, GM with some notion of similarity (maybe NBLAST and or spec sim?), GM with multigraph, GM with multigraph + similarity.
    - (Maybe) Seeded graph matching with the known pairs as seeds? I actually use this in my work... so it is useful.
    - Some interesting inference about pairedness? Would be nice to show what this can be used for, or demonstrate which parts are more/less bilaterally similar?
    - Maybe something about testing for homotypic connections? 
- Bilateral symmetry/testing
    - How similar are the SBM models?
        - chi square test?
    - How similar are the RDPG models (nonpar/semipar)?
        - maybe select best model and compare left right on those
    - Can we say anything about the correlation under these different models?
    - Testing homotopic affinity (by edge type)
- Directedness: testing for whether the graph or specific parts of it are meaningfully directed. (Do we know how to do this?)
- Feedforwardness: describing (and hopefully modeling) a feedforward pathway through the network, expanding to include multinetwork models. 
    - Some description of the chain predicted by signal flow or cascades or graph match flow etc.
    - Comparisons of the flows for different network types (e.g. AA, AD, etc.) 
    - Testing for feedforwardness with spring rank model
    - maybe goes in Cambridge paper?

## Code
- [Flow/hierarchy/ranking](https://github.com/microsoft/graspologic/issues/636) into graspologic
- Improve the estimation code to make it easier to fit to data in a useful way, examine the models, etc. (as necessary)
- Adjacency with dendrogram for hierarchical clustering that is not complete
    - I have code, not pretty, probably not generalizable yet
- [Tests from statistical connectomics](https://github.com/microsoft/graspologic/issues/570) into graspologic
    - We should talk to Eric/decide what we actually want first
- Bar dendrogram plotting in general