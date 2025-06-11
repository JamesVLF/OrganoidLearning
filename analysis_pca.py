

import numpy as np
import matplotlib.pyplot as plt

def eigenvalues_eigenvectors(matrix):
    W, U = np.linalg.eigh(matrix)
    rank = min(*matrix.shape)
    U = U[:, -rank:]
    sgn = (-1)**(U[0, :] < 0)
    return W[-rank:][::-1], (U * sgn[np.newaxis, :])[:, ::-1]

def plot_evectmatrix(corr, sttc):
    fig, plot = plt.subplot_mosaic("AB", figsize=(14, 7))
    Wcorr, Ucorr = eigenvalues_eigenvectors(corr)
    Wsttc, Usttc = eigenvalues_eigenvectors(sttc)

    im1 = plot["A"].imshow(Ucorr.T, interpolation='none', cmap="magma")
    plot["A"].set(title='Eigenvectors of Correlation', xlabel='Neuron', ylabel='Component')
    fig.colorbar(im1, ax=plot["A"], shrink=0.7)

    im2 = plot["B"].imshow(Usttc.T, interpolation='none', cmap="magma")
    plot["B"].set(title='Eigenvectors of STTC', xlabel='Neuron', ylabel='Component')
    fig.colorbar(im2, ax=plot["B"], shrink=0.7)

    plt.tight_layout()
    plt.show()

def plot_basis(U, method):
    plt.figure(figsize=(6, 8))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.stem(U[:, i])
        if i < 4:
            plt.xticks([])
    plt.xlabel('Neuron Index')
    plt.suptitle(f'First 5 Eigenvectors of {method}', y=0.92)
    plt.tight_layout()

def PCAplots(Ucorr, Usttc):
    plt.figure(figsize=(7, 5))
    plt.scatter(Ucorr[:, 0], Ucorr[:, 1], label='Correlation')
    plt.scatter(Usttc[:, 0], Usttc[:, 1], label='STTC')
    plt.legend()
    plt.title("PCA: First Two Eigenvectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def reconstruct(W, U, rank):
    Wd = np.diag(W[:rank])
    Ur = U[:, :rank]
    return Ur @ Wd @ Ur.T

def reconstruction_errors(A, ranks):
    norm = np.linalg.norm(A)
    W, U = eigenvalues_eigenvectors(A)
    return np.array([
        np.linalg.norm(reconstruct(W, U, r) - A) / norm for r in ranks
    ])

def ReconstructPlots(Wcorr, Wsttc, Corr, STTC):
    fig, plot = plt.subplot_mosaic("AB", figsize=(15, 5))

    index = np.arange(1, len(Wcorr) + 1)
    plot["A"].semilogy(index, Wcorr, label='Correlation')
    plot["A"].plot(index, Wsttc, label='STTC')
    plot["A"].set(title='Eigenvalue Spectrum', xlabel='Index', ylabel='Eigenvalue')
    plot["A"].legend()

    errs_corr = reconstruction_errors(Corr, range(len(Corr)))
    errs_sttc = reconstruction_errors(STTC, range(len(STTC)))

    plot["B"].plot(errs_corr, label='Correlation Matrix')
    plot["B"].plot(errs_sttc, label='STTC Matrix')
    plot["B"].set(title='Reconstruction Error', xlabel='Components', ylabel='Relative Error')
    plot["B"].legend()

    plt.tight_layout()
    plt.show()

def EigenvectorAnalysis(Ucorr, Usttc):
    figLayout = """
                AABB
                CCDD
                CCDD
                """
    fig, plot = plt.subplot_mosaic(figLayout, figsize=(16, 8))
    fig.tight_layout(pad=5.0)

    plot["A"].stem(Ucorr[:, 0])
    plot["A"].set_title('First Eigenvector of Correlation')

    plot["B"].stem(Usttc[:, 0])
    plot["B"].set_title('First Eigenvector of STTC')

    imC = plot["C"].imshow(Ucorr.T, cmap="magma")
    plot["C"].set_title("Eigenvectors of Correlation")
    fig.colorbar(imC, ax=plot["C"], shrink=0.7)

    imD = plot["D"].imshow(Usttc.T, cmap="magma")
    plot["D"].set_title("Eigenvectors of STTC")
    fig.colorbar(imD, ax=plot["D"], shrink=0.7)

    plt.show()

def evectLayout(neuron_coords, vect, sttc_matrix, threshold=0, cmap="magma"):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_facecolor("lightgrey")
    x, y = neuron_coords[:, 0], neuron_coords[:, 1]
    sc = ax.scatter(x, y, c=vect, cmap=cmap, s=100)
    plt.colorbar(sc)

    N = len(x)
    for i in range(N):
        for j in range(i + 1, N):
            if sttc_matrix[i, j] > threshold:
                plt.plot([x[i], x[j]], [y[i], y[j]], linewidth=sttc_matrix[i, j], color='black')

    plt.title("Eigenvector-Weighted Neuron Layout")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
