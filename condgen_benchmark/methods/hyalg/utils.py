def leave_two_out_sum(M, dim=1):
    M.fill_diagonal_(0)
    return M.sum(dim=dim, keepdim=True) - M


def indicator_matrix(Y):
    return (Y[:, None] <= Y[None, :]).float()


def enforce_positive_first_nonzero(theta):
    for i in range(len(theta)):
        if theta[i] != 0:
            return theta if theta[i] > 0 else -theta
    return theta
