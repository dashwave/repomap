from collections import Counter, defaultdict, namedtuple
import sqlite3
import math
from pathlib import Path
from tqdm import tqdm
import logging
import os
from repomap.vendor.grep_ast import TreeContext, filename_to_lang
from tree_sitter_language_pack import get_language, get_parser  # noqa: E402
from pygments.lexers import guess_lexer_for_filename
from importlib import resources
from pygments.token import Token
import time
import tiktoken
from typing import List, Tuple
from tree_sitter import Node
import json
import networkx as nx


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("get-repo-map")
SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)
Tag = namedtuple("Tag", "rel_fname fname line name kind".split())
encoding = tiktoken.get_encoding("cl100k_base")

def get_scm_fname(lang):
    # Load the tags queries
    try:
        return resources.files(__package__).joinpath("queries", f"tree-sitter-{lang}-tags.scm")
    except KeyError:
        return


def print_tree(node, indent=0):
    # # """
    # # Recursively prints the Tree-sitter AST starting at 'node'.
    
    # # Each node is printed with indentation proportional to its depth.
    # # If the node has associated text (e.g. an identifier), that text is printed,
    # # truncated to a reasonable length.
    # # """
    # indent_str = "  " * indent
    # # Attempt to get the node's text if available.
    # try:
    #     # node.text may be a byte string; decode it if needed.
    #     text = node.text.decode("utf8") if isinstance(node.text, bytes) else node.text
    #     text = text.strip()
    # except Exception:
    #     text = ""
    
    # # Optionally, truncate text if it's too long.
    # if len(text) > 40:
    #     text = text[:37] + "..."
    
    # # Print node type (and text if present)
    # if text:
    #     print(f"{indent_str}{node.type}: {text}")
    # else:
    #     print(f"{indent_str}{node.type}")
    
    # # Recursively print each child node.
    # for child in node.children:
    #     print_tree(child, indent + 1)
    ast_dict = node_to_dict(node)
    print(json.dumps(ast_dict, indent=2))

def node_to_dict(node):
    """
    Recursively converts a Tree-sitter node into a dictionary.
    """
    node_dict = {
        "type": node.type,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "start_point": node.start_point,  # (row, column)
        "end_point": node.end_point,      # (row, column)
    }
    
    # Recursively process children if they exist.
    children = [node_to_dict(child) for child in node.children]
    if children:
        node_dict["children"] = children
        for i, child in enumerate(children):
            node_dict["children"][i]["field_name"] = node.field_name_for_child(i)
    try:
        text = node.text
        if isinstance(text, bytes):
            text = text.decode("utf8")
        text = text.strip()
        if text:
            node_dict["text"] = text
    except Exception:
        pass
    
    return node_dict
class RepoMap:
    warned_files = set()

    def __init__(self,
            root,
            verbose=False,
            max_map_tokens=20000,
            map_mul_no_files=0.5,
            max_context_window=2000000
        ):
        self.root = root
        self.max_map_tokens = max_map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window
        self.verbose = verbose
        self.tree_context_cache = {}

        logger.info(f"RepoMap initialized with root: {self.root}")

    def token_count(self, text):
        num_tokens = len(encoding.encode(text))
        return num_tokens
    
    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens, mentioned_fnames, mentioned_idents
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.last_map = result

        return result

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        """Generate a map of the repository's files and their contents.
        
        Args:
            chat_files: List of files currently in the chat
            other_files: List of other files in the repo to potentially include
            mentioned_fnames: Set of filenames mentioned in the conversation
            mentioned_idents: Set of identifiers mentioned in the conversation
            
        Returns:
            A string containing the formatted repo map, or None if no map could be generated
        """
        # Early returns if we can't/shouldn't generate a map
        if not other_files:
            return None
            
        mentioned_fnames = mentioned_fnames or set()
        mentioned_idents = mentioned_idents or set()

        # Calculate max tokens to use
        max_map_tokens = self.max_map_tokens
        if not chat_files and self.max_context_window:
            padding = 4096
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
            if target > 0:
                max_map_tokens = target

        # Generate the file listing
        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh=False
            )
        except RecursionError:
            logger.error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return None
        

        if not files_listing:
            return None

        # Log token count if verbose
        if self.verbose:
            num_tokens = self.token_count(files_listing)

        # Format the final output
        other = "other " if chat_files else ""
        prefix = f"Repo-map: {other}"
        
        return prefix + files_listing

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            return fname

    def get_ranked_tags(
        self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents, progress=None
    ):
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)
        imports = defaultdict(set)
        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        for fname in fnames:    
            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    logger.warning(f"Repo-map can't include {fname}")
                    logger.warning(
                        "Has it been deleted from the file system but not from git?"
                    )
                    self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            logger.debug(f"tags for {rel_fname}:")
            for tag in tags:
                logger.debug(f"  {tag}")
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

                elif tag.kind == "import":
                    imports[tag.name].add(rel_fname)
                    key = (tag.name, "import")
                    definitions[key].add(tag)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)
        # logger.debug(f"defines: {defines}")
        # logger.debug(f"references: {references}")
        # logger.debug(f"personalization: {personalization}")

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        for ident in idents:
            definers = defines[ident]
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"):
                mul = 0.1
            else:
                mul = 1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)
        

        for import_name, fnames in imports.items():
            for fname in fnames:
                print(f"adding edge for import: {fname} -> {import_name}")
                G.add_edge(fname, import_name, weight=10, ident="import")

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            # Issue #1536
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        # print(f"G.nodes: {G.nodes}")
        # print(f"Ranked: {ranked}")
        for src in G.nodes:
            if progress:
                progress()

            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        print(f"ranked_definitions: {ranked_definitions}")

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            # print(f"{rank:.03f} {fname} {ident}")
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(self.get_rel_fname(fname) for fname in other_fnames)

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags
    
    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            logger.warning(f"File not found error: {fname}")
            return None
    
    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []
        try:
            data = list(self.get_tags_raw(fname, rel_fname))
        except Exception as err:
            logger.warning(f"Error getting tags for {fname}: {err}")
            self.warned_files.add(fname)
            return []
        return data

    def get_tags_raw(self, fname, rel_fname):
        file_extension = os.path.splitext(fname)[1]
        if file_extension == ".dart":
            lang = "dart"
        else:
            lang = filename_to_lang(fname)
        logger.info(f"lang: {lang}")
        if not lang:
            return

        try:
            # init language
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            logger.warning(f"Skipping file {rel_fname}: {err}")
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()
        code = Path(fname).read_text()
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)        
        captures_list: List[Tuple[str, List[Node]]] = []

        for k,v in captures.items():
            for node in v:
                captures_list.append((k, node))
        saw = set()
        for tag, node in captures_list:
            # logger.info(f"tag: {tag}, node: {node}, {type(node)}")
            # logger.info(f"node: {node.text.decode('utf-8')}")
            if tag.startswith("name.definition"):
                kind = "def"
                name = node.text.decode("utf-8")
            elif tag.startswith("name.reference"):
                kind = "ref"
                name = node.text.decode("utf-8")
            elif tag.startswith("name.import"):
                kind = "import"
                name = node.text.decode("utf-8")
                # remove the quotes
                name = name.strip("'")
                if not name.startswith("package:"):
                    # if the name is not a package name, it is a relative path
                    # so we need to join it with the directory of the current file
                    # to get the full path
                    name = os.path.normpath(os.path.join(os.path.dirname(rel_fname), name))
                else:
                    continue
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=name,
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:  # On Windows, bad ref to time.clock which is deprecated?
            # self.io.tool_error(f"Error lexing {fname}")
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            logger.info(f"yielding token from lexer: {token}")
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        """Generate an uncached map of the repository's structure and content.
        
        This function analyzes the repository's files to create a structured view that:
        1. Prioritizes important files and mentioned identifiers
        2. Optimizes content to fit within token limits
        3. Creates a hierarchical view of code definitions and references
        
        Args:
            chat_fnames: List of files currently in the chat
            other_fnames: List of other repository files to analyze
            max_map_tokens: Maximum tokens allowed in the output
            mentioned_fnames: Set of filenames mentioned in conversation
            mentioned_idents: Set of code identifiers mentioned in conversation
            
        Returns:
            A formatted string containing the repository map with relevant code sections
        """
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
        )

        logger.info(f"ranked_tags:")
        for tag in ranked_tags:
            logger.info(f"  ranked_tag: {tag}")


        other_rel_fnames = sorted(set(self.get_rel_fname(fname) for fname in other_fnames))
        # special_fnames = filter_important_files(other_rel_fnames)
        special_fnames = []
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(int(max_map_tokens // 25), num_tags)
        while lower_bound <= upper_bound:
            # dump(lower_bound, middle, upper_bound)
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.05
            if (num_tokens <= max_map_tokens and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = int((lower_bound + upper_bound) // 2)

        return best_tree

    tree_cache = dict()

    def to_tree(self, tags, chat_rel_fnames):
        logger.info(f"to_tree tags: {tags}")
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            if this_rel_fname in chat_rel_fnames:
                continue

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    print(f"calling render_tree for {cur_fname}: {lois}")
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output
    
    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)

        context: TreeContext = None
        if (
            rel_fname not in self.tree_context_cache
            or self.tree_context_cache[rel_fname]["mtime"] != mtime
        ):
            code = Path(abs_fname).read_text() or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=True,
                line_number=True,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                # header_max=30,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        logger.info(f"lois for {rel_fname}: {lois}")
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        return res

    def get_warned_files(self):
        return self.warned_files
