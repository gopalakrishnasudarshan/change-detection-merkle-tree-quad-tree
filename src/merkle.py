import hashlib

def build_merkle_tree(leaf_hashes: list[bytes]) -> list[list[bytes]]:
    if not leaf_hashes:
        raise ValueError("Cannot build Merkle tree from empty leaf list")
    
    levels: list[list[bytes]] = [leaf_hashes]

    while len(levels[-1]) > 1:
        current = levels[-1]
        next_level: list[bytes] = []

        i = 0 
        while i < len(current):
            left = current[i]
            if i + 1 < len(current):
                right = current[i+1]
            else:
                right = left
        
            parent = hashlib.sha256(left + right).digest()
            next_level.append(parent)
            i += 2
    
        levels.append(next_level)

    return levels

def merkle_diff_changed_leaves(tree1: list[list[bytes]], tree2: list[list[bytes]]) -> list[int]:
    if len(tree1)!= len(tree2):
        raise ValueError("Tree level count mismatch")
    
    top_level = len(tree1) - 1
    if len(tree1[top_level]) != 1 or len(tree2[top_level]) != 1:
        raise ValueError("Expected a single root at the top level")
    
    changed_set: set[int] =set()
    stack: list[tuple[int,int]] = [(top_level,0)]
    
    while stack:
        level, idx = stack.pop()
        
        if tree1[level][idx] == tree2[level][idx]:
            continue
        
        if level == 0:
            changed_set.add(idx)
            continue
        
        child_level = level - 1
        left = 2 * idx
        right = 2 * idx + 1
        n = len(tree1[child_level])
        
        if left < n:
            stack.append((child_level, left))
        if right <  n:
            stack.append((child_level,right))
       
      
    changed = sorted(changed_set)
    return changed
    