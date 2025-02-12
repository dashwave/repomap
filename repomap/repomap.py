from collections import Counter, defaultdict, namedtuple
import sqlite3
import math
from pathlib import Path
from tqdm import tqdm
import logging
import os
from grep_ast import TreeContext, filename_to_lang
from tree_sitter_languages import get_language, get_parser  # noqa: E402
from pygments.lexers import guess_lexer_for_filename
from importlib import resources
from pygments.token import Token
import time
import tree_sitter_languages
import tiktoken

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


class RepoMap:
    warned_files = set()

    def __init__(self, root, verbose=False):
        self.root = root
        self.max_map_tokens = 4096
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
        
        logger.info(f"files_listing: <{files_listing}>")

        if not files_listing:
            return None

        # Log token count if verbose
        if self.verbose:
            num_tokens = self.token_count(files_listing)
            logger.info(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

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
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

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
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        logger.info(f"defines: {defines}")
        logger.info(f"references: {references}")
        logger.info(f"definitions: {definitions}")


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
        data = list(self.get_tags_raw(fname, rel_fname))
        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
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

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
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


        other_rel_fnames = sorted(set(self.get_rel_fname(fname) for fname in other_fnames))
        # special_fnames = filter_important_files(other_rel_fnames)
        special_fnames = []
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags
        logger.info(f"ranked_tags: {ranked_tags}")

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
            logger.info(f"lower_bound: {lower_bound}, middle: {middle}, upper_bound: {upper_bound}")
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)
            logger.info(f"tree: {tree}")
            logger.info(f"num_tokens: {num_tokens}")

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
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
        if not tags:
            return ""

        logger.debug(f"calling to_tree for tags: {tags} and chat_rel_fnames: {chat_rel_fnames}")
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
                color=False,
                line_number=False,
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
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        return res