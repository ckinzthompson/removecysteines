import torch
import esm
import numpy as np
import sys
import time
import argparse

if torch.cuda.is_available():
	device = 'cuda'
elif torch.backends.mps.is_available():
	device = 'mps'
else:
	device = 'cpu'

letters = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
letterids = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

def _embed_sequences(data,model,batch_converter):
	batch_labels, batch_strs, batch_tokens = batch_converter(data)

	## note: encoded sequence has a start and end position, so length is two longer than sequence. removing
	batch_tokens = batch_tokens.to(device)
	with torch.no_grad():
		results = model(batch_tokens, repr_layers=[model.num_layers,], return_contacts=False)

	reps = results['representations'][model.num_layers].cpu().numpy()
	logits = results['logits'].cpu().numpy()

	if device == 'mps':
		torch.mps.empty_cache()

	return reps,logits

def embed_sequences(data,model,batch_converter):
	'''
	Note: on MPS it's essential to iterate rather than batch, b/c of memory pressure issues. I noticed completely wrong values popping up b/c of swapping (I think).
	external validation here: https://huggingface.co/docs/diffusers/en/optimization/mps
	'''

	if device == 'mps':
		reps,logits = _embed_sequences([data[0],],model,batch_converter)

		for i in range(1,len(data)):
			_reps,_logits = _embed_sequences([data[i],],model,batch_converter)
			reps = np.concatenate((reps,_reps),axis=0)
			logits = np.concatenate((logits,_logits),axis=0)
	else:
		reps,logits = _embed_sequences(data,model,batch_converter)
	return reps,logits

def generate_pointmutants(sequence,index):
	data = []
	for letter in letters:
		mutated_sequence = sequence[:index] + letter + sequence[index+1:]
		data.append(('%d%s'%(index,letter),mutated_sequence))
	return data
	
def calc_pseudoperplexity(logits,seq):
	#### eqn 4
	#### logits (seq,latent)
	#### sequence (seq)
	
	#### calculate probabilities
	probs = np.exp(logits)
	probs /= np.sum(probs,axis=1)[:,None]
	
	## decode sequence
	seq_ids = np.array([letterids[letters.index(seq[i])] for i in range(len(seq))])
	nlp = -np.log(probs[1:-1,seq_ids]) ## remove CLS and EOS tokens.

	## calculate pseudoperplexity
	pppl = np.exp(np.mean(nlp))
	return pppl

def calc_pseudoperplexities(logits,data):
	perp = np.array([calc_pseudoperplexity(logits[i],data[i][1]) for i in range(len(data))])
	return perp

def calc_given_best(sequence,model,batch_converter,indices):
	from scipy.special import softmax

	data = [(0,sequence)]
	reps, logits = embed_sequences(data,model,batch_converter)
	
	out = []
	for index in indices:
		p = softmax(logits[0,index+1,letterids])
		p[letters.index(sequence[index])] = 0.
		p /= p.sum()
		out.append([sequence[index],index,letters[p.argmax()]])
	return out


def main(wt_sequence, n_rounds, show_pca):
	
	#### Load ESM-2 model
	model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
	# model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
	print('ESM2-t%s'%(model.num_layers))

	#### Find GPU type

	print('Using Device:',device)
	print('----------')

	batch_converter = alphabet.get_batch_converter()
	model.eval()  # disables dropout for deterministic results
	model = model.to(device) # put onto the gpu

	# ## Exclude certain letters from consideration
	# letters = [ll for ll in list(alphabet.tok_to_idx.keys()) if ll.isalpha()]	
	# letters = [ll for ll in letters if ll not in ['B','J','O','U','X','Z']]
	# letterids = np.array([alphabet.tok_to_idx[ll] for ll in letters])

	#### Reporting Statistics
	print('WT Sequence: %s'%(wt_sequence))
	print('Length: %d'%(len(wt_sequence)))

	ncys = wt_sequence.count('C')
	print('Num. Cys: %d'%(ncys))
	if ncys == 0:
		print('No Cys to remove. Finished!')
		sys.exit(0)

	print('Cys locations:',*[index for index in range(len(wt_sequence)) if wt_sequence[index] == 'C'])
	print('\n---------- Optimization ----------')	
	reps,logits = embed_sequences([('wt',wt_sequence),],model,batch_converter)
	wt_pp = calc_pseudoperplexity(logits[0],wt_sequence)
	print('0. WT Perplexity: %.2f'%(wt_pp))

	## Design
	mut_sequence = ''.join(list(wt_sequence)) ## make a deep copy
	indices = np.array([index for index in range(len(mut_sequence)) if mut_sequence[index] == 'C'])

	#### Step 1. Remove all C using the best (unmasked) alternative
	mutations = calc_given_best(mut_sequence,model,batch_converter,indices)
	for mutation in mutations:
		orig,ind,repl = mutation
		mut_sequence = mut_sequence[:ind] + repl + mut_sequence[ind+1:]

	reps,logits = embed_sequences([('mut',mut_sequence),],model,batch_converter)
	mut_pp = calc_pseudoperplexity(logits[0],mut_sequence)
	print('1. Initial MUT perplexity: %.2f'%(mut_pp))
	for mutation in mutations:
		print('\tC%d%s'%(mutation[1],mutation[2]))

	#### Step 2. Scan all point changes to maximize perplexity
	cls = {}
	for iter in range(n_rounds):
		## get starting point
		reps,logits = embed_sequences([('mut',mut_sequence),],model,batch_converter)
		mut_pp = calc_pseudoperplexity(logits[0],mut_sequence)
		best = [-1,mut_pp,-1]

		t0 = time.time()
		for index in indices:
			data = generate_pointmutants(mut_sequence,index)
			data = data[:-1] # no C
			t0 = time.time()
			reps,logits = embed_sequences(data,model,batch_converter)
			t1 = time.time()
			pp = calc_pseudoperplexities(logits,data)
			if pp.max() > best[1]:
				best = [index,pp.max(),pp.argmax()]
			
			if show_pca:
				for i in range(len(data)):
					if not data[i][1] in cls:
						cls[data[i][1]] = reps[i,0].copy()
		
		print('2.%d Polish MUT perplexity %.2f'%(iter,best[1]))
		# print('\tTime:',t1-t0,(t1-t0)/(19*len(indices)))
		if 	best[0] != -1:
			print('\tC%d%s'%(best[0],letters[best[2]]))
			mut_sequence = mut_sequence[:best[0]] + letters[best[2]] + mut_sequence[best[0]+1:]
		else:
			print('\tNo better change')
			break

	## Step 3. Finish up
	print('\n---------- Final ----------')	
	print('MUT Sequence: %s'%(mut_sequence))
	reps,logits = embed_sequences([('mut',mut_sequence),],model,batch_converter)
	mut_pp = calc_pseudoperplexity(logits[0],mut_sequence)
	print('MUT Perplexity: %.2f'%(mut_pp))
	print('Mutations:')
	for index in indices:
		print('\t%s%d%s'%(wt_sequence[index],index,mut_sequence[index]))


	### Step 4. Analysis		
	if show_pca:
		q = np.array([cls[k] for k in cls.keys()])
		import matplotlib.pyplot as plt
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		w = pca.fit_transform(q)
		plt.plot(w[:,0],w[:,1],'o',color='gray',label='Point mutants')

		reps,logits = embed_sequences([('wt',wt_sequence),],model,batch_converter)
		ww = pca.transform(reps[0,0][None,:])[0]
		plt.plot(ww[0],ww[1],'o',color='tab:blue',label='WT')

		reps,logits = embed_sequences([('mut',mut_sequence),],model,batch_converter)
		ww = pca.transform(reps[0,0][None,:])[0]
		plt.plot(ww[0],ww[1],'o',color='tab:red',label='Final MUT')

		plt.xlabel('PCA1')
		plt.ylabel('PCA2')
		plt.legend()
		plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Remove Cysteines")
	parser.add_argument("sequence", type=str, help="WT protein sequence to alter")
	parser.add_argument("-n", "--n_rounds", type=int, default=20, help="Maximum Number of Polishing Rounds")
	parser.add_argument("-p", "--pca", type=bool, default=False, help="Show embedding PCA?")
	
	args = parser.parse_args()
	main(args.sequence,args.n_rounds,args.pca)